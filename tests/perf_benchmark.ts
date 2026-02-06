// tests/perf_benchmark.ts
/**
 * 性能基准测试（固定 N=100000）：
 * - 插入耗时
 * - 搜索吞吐与延迟
 * - save/load 耗时
 * - 内存占用（Node RSS/heap/external/arrayBuffers）+ 文件大小（vectors/meta/dump）
 *
 * 运行：
 *   npx ts-node tests/perf_benchmark.ts
 *
 * 建议（可选）：
 *   node --expose-gc node_modules/.bin/ts-node tests/perf_benchmark.ts
 *
 * 可选环境变量（仍可调参）：
 *   PERF_DIM=384
 *   PERF_K=10
 *   PERF_QUERIES=200
 *   PERF_BATCH=500
 *   PERF_REPORT=1
 */

import fs from "fs";
import path from "path";
import os from "os";
import { MiniVectorDB } from "../src/index";

type StageStats = {
	name: string;
	ms: number;
	mem: NodeJS.MemoryUsage;
	extra?: Record<string, any>;
};

const N = 100_000;

function nowMs(): number {
	const [s, ns] = process.hrtime();
	return s * 1000 + ns / 1e6;
}

function fmtBytes(n: number) {
	const units = ["B", "KB", "MB", "GB", "TB"];
	let i = 0;
	let x = n;
	while (x >= 1024 && i < units.length - 1) {
		x /= 1024;
		i++;
	}
	return `${x.toFixed(2)} ${units[i]}`;
}

function snapshotMem(): NodeJS.MemoryUsage {
	return process.memoryUsage();
}

function memSummary(m: NodeJS.MemoryUsage) {
	return {
		rss: fmtBytes(m.rss),
		heapUsed: fmtBytes(m.heapUsed),
		heapTotal: fmtBytes(m.heapTotal),
		external: fmtBytes(m.external),
		arrayBuffers: fmtBytes(m.arrayBuffers),
	};
}

function safeStatSize(file: string): number {
	try {
		return fs.statSync(file).size;
	} catch {
		return 0;
	}
}

// 稳定可复现实验向量
function genVec(dim: number, seed: number): Float32Array {
	const v = new Float32Array(dim);
	let x = seed | 0;
	for (let i = 0; i < dim; i++) {
		x = (x * 1664525 + 1013904223) | 0;
		const r = (x >>> 0) / 4294967296;
		const t = r * 2 - 1;
		v[i] = Math.max(-1, Math.min(1, t * 0.9 + (i % 7) * 0.01));
	}
	return v;
}

function pickIds(total: number, count: number): number[] {
	const out: number[] = [];
	let x = 123456789;
	for (let i = 0; i < count; i++) {
		x = (x * 1103515245 + 12345) | 0;
		out.push(Math.abs(x) % total);
	}
	return out;
}

async function stage<T>(
	name: string,
	fn: () => Promise<T>,
	extra?: Record<string, any>,
): Promise<{ result: T; stat: StageStats }> {
	if ((global as any).gc) (global as any).gc();
	const mem0 = snapshotMem();
	const t0 = nowMs();
	const result = await fn();
	const t1 = nowMs();
	if ((global as any).gc) (global as any).gc();
	const mem1 = snapshotMem();
	return {
		result,
		stat: {
			name,
			ms: Math.round(t1 - t0),
			mem: mem1,
			extra: { ...extra, mem0: memSummary(mem0), mem1: memSummary(mem1) },
		},
	};
}

async function main() {
	const dim = parseInt(process.env.PERF_DIM || "384", 10);
	const k = parseInt(process.env.PERF_K || "10", 10);
	const queryCount = parseInt(process.env.PERF_QUERIES || "200", 10);

	// Windows 上 batch=2000 并发会导致抖动/峰值很高，建议默认更保守
	const batch = parseInt(process.env.PERF_BATCH || "500", 10);

	const report = process.env.PERF_REPORT === "1";

	console.log("========================================");
	console.log("      MINIVECTORDB PERF BENCHMARK       ");
	console.log("========================================");
	console.log(`N=${N}`);
	console.log(`dim=${dim}, k=${k}, queries=${queryCount}, batch=${batch}`);
	console.log(
		`node=${process.version} platform=${process.platform} cpu=${os.cpus()?.[0]?.model ?? "unknown"}`,
	);
	console.log(
		`GC available: ${(global as any).gc ? "YES (node --expose-gc)" : "NO"}`,
	);
	console.log("");

	const baseDir = path.join(__dirname, "../data/perf");
	fs.mkdirSync(baseDir, { recursive: true });

	const caseDir = path.join(baseDir, `n_${N}`);
	fs.rmSync(caseDir, { recursive: true, force: true });
	fs.mkdirSync(caseDir, { recursive: true });

	const dumpPath = path.join(caseDir, "dump.bin");
	const metaPath = path.join(caseDir, "metadata.json");
	const vecPath = path.join(caseDir, "vectors.f32.bin");

	// 关键：capacity 必须 >= N（建议留 buffer）
	const capacity = Math.floor(N * 1.1);

	const db = new MiniVectorDB({
		dim,
		metaDbPath: metaPath,
		vectorStorePath: vecPath,
		rerankMultiplier: 30,
		capacity,
	});

	const initRes = await stage("init()", async () => {
		await db.init();
		return true;
	});

	let inserted = 0;
	const insRes = await stage(
		`insert(${N})`,
		async () => {
			while (inserted < N) {
				const end = Math.min(N, inserted + batch);

				// ✅ 为了更稳定的 benchmark：串行/小并发更可靠
				// 这里用 Promise.all(batch) 但 batch 默认较小
				const tasks: Promise<void>[] = [];
				for (let i = inserted; i < end; i++) {
					const v = genVec(dim, i + 1);
					tasks.push(
						db.insert({
							id: `doc-${i}`,
							vector: v,
							metadata: { i },
						}),
					);
				}
				await Promise.all(tasks);
				inserted = end;

				if (inserted % (batch * 10) === 0 || inserted === N) {
					process.stdout.write(
						`  inserted ${inserted}/${N} (${((inserted / N) * 100).toFixed(1)}%)\r`,
					);
				}
			}
			process.stdout.write("\n");
			return true;
		},
		{ batch },
	);

	const statsAfterInsert = db.getStats();
	const vecSize = safeStatSize(vecPath);
	const metaSize = safeStatSize(metaPath);

	console.log("After insert:");
	console.log("  dbStats:", statsAfterInsert);
	console.log("  vecFile:", fmtBytes(vecSize));
	console.log("  metaFile:", fmtBytes(metaSize));

	const qIds = pickIds(N, queryCount);

	const searchRes = await stage(
		`search(${queryCount} queries)`,
		async () => {
			let totalMs = 0;
			let minMs = Number.POSITIVE_INFINITY;
			let maxMs = 0;

			for (let i = 0; i < qIds.length; i++) {
				const id = qIds[i];
				const q = genVec(dim, id + 1);

				const t0 = nowMs();
				const r = await db.search(q, k);
				const t1 = nowMs();

				const dt = t1 - t0;
				totalMs += dt;
				if (dt < minMs) minMs = dt;
				if (dt > maxMs) maxMs = dt;

				if (r.length === 0) throw new Error(`Empty result at query #${i}`);
			}

			const avg = totalMs / qIds.length;
			return { avgMs: avg, minMs, maxMs, qps: 1000 / avg };
		},
		{ k, queryCount },
	);

	console.log("Search summary:", searchRes.result);

	const saveRes = await stage("save(dump.bin)", async () => {
		await db.save(dumpPath);
		return true;
	});

	const dumpSize = safeStatSize(dumpPath);
	console.log("Dump file:", fmtBytes(dumpSize));

	const closeRes = await stage("close()", async () => {
		await db.close();
		return true;
	});

	// reload instance
	const db2 = new MiniVectorDB({
		dim,
		metaDbPath: metaPath,
		vectorStorePath: vecPath,
		rerankMultiplier: 30,
		capacity,
	});

	const init2Res = await stage("init() [reload instance]", async () => {
		await db2.init();
		return true;
	});

	const loadRes = await stage("load(dump.bin)", async () => {
		await db2.load(dumpPath);
		return true;
	});

	const search2Res = await stage(
		`search(${queryCount} queries) [after load]`,
		async () => {
			let totalMs = 0;
			let minMs = Number.POSITIVE_INFINITY;
			let maxMs = 0;

			for (let i = 0; i < qIds.length; i++) {
				const id = qIds[i];
				const q = genVec(dim, id + 1);

				const t0 = nowMs();
				const r = await db2.search(q, k);
				const t1 = nowMs();

				const dt = t1 - t0;
				totalMs += dt;
				if (dt < minMs) minMs = dt;
				if (dt > maxMs) maxMs = dt;

				if (r.length === 0)
					throw new Error(`Empty result after load at query #${i}`);
			}

			const avg = totalMs / qIds.length;
			return { avgMs: avg, minMs, maxMs, qps: 1000 / avg };
		},
		{ k, queryCount },
	);

	const close2Res = await stage("close() [reload instance]", async () => {
		await db2.close();
		return true;
	});

	const record = {
		N,
		dim,
		k,
		queryCount,
		batch,
		capacity,
		files: {
			dumpPath,
			metaPath,
			vecPath,
			dumpBytes: dumpSize,
			metaBytes: metaSize,
			vecBytes: vecSize,
		},
		stages: [
			initRes.stat,
			insRes.stat,
			searchRes.stat,
			saveRes.stat,
			closeRes.stat,
			init2Res.stat,
			loadRes.stat,
			search2Res.stat,
			close2Res.stat,
		],
	};

	const reportPath = path.join(baseDir, `perf_report_${N}_${Date.now()}.json`);
	fs.writeFileSync(reportPath, JSON.stringify(record, null, 2), "utf-8");

	console.log("\nCASE SUMMARY:");
	for (const s of record.stages) {
		const rss = (s.extra?.mem1?.rss as string) || "";
		console.log(`  - ${s.name}: ${s.ms} ms${rss ? ` | rss=${rss}` : ""}`);
	}
	console.log("");
	console.log("========================================");
	console.log("DONE.");
	console.log("report:", reportPath);

	if (report) {
		console.log("\n--- RAW REPORT ---");
		console.log(JSON.stringify(record, null, 2));
	}
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
