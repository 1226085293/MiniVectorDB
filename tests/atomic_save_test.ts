// tests/atomic_save_test.ts
import { MiniVectorDB } from "../src/index";
import os from "os";
import path from "path";
import fs from "fs";
import { assert } from "./_util";

function delay(ms: number) {
	return new Promise((r) => setTimeout(r, ms));
}

function statSize(p: string): number {
	try {
		return fs.statSync(p).size;
	} catch {
		return 0;
	}
}

async function reopen(storageDir: string) {
	return await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity: 50_000,
		preloadVectors: false,
	});
}

async function main() {
	console.log("--- ATOMIC SAVE TEST START (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-atomic-"),
	);

	// 这些文件名按你项目当前约定；若未来改名也不影响主要逻辑（只影响 size check）
	const dumpPath = path.join(storageDir, "dump.bin");
	const metaPath = path.join(storageDir, "metadata.json");
	const vecPath = path.join(storageDir, "vectors.f32.bin");

	let db = await reopen(storageDir);
	const dim = db.getStats().dim;
	assert(dim === 384, `expected dim=384, got=${dim}`);

	// 先写点数据
	const baseVec = new Array(dim).fill(0).map((_, i) => i / dim);
	await db.insert({ id: "atomic-0", input: baseVec, metadata: { round: 0 } });
	await db.save();
	await db.close();

	// 多轮：每轮都确保 save 后可 reopen 且能搜到预期 ID
	const ROUNDS = 8;

	for (let round = 1; round <= ROUNDS; round++) {
		db = await reopen(storageDir);

		// 插入一个更“独特”的向量，降低碰撞/平局干扰
		const v = new Array(dim).fill(0).map((_, i) => (i + round) / (dim + 13));
		const id = `atomic-${round}`;

		await db.insert({ id, input: v, metadata: { round } });

		await db.save();

		// 给 Windows 一点喘息：避免文件句柄未释放导致下一步 reopen 偶发 EPERM
		await db.close();
		await delay(50);

		// 基础完整性：文件存在且不为 0（防止短写/截断）
		assert(
			statSize(dumpPath) > 0,
			`dump.bin should exist and be >0 (round=${round})`,
		);
		// meta/vec 在不同实现下可能可选，但通常应该存在
		assert(
			statSize(metaPath) > 0,
			`metadata.json should exist and be >0 (round=${round})`,
		);
		assert(
			statSize(vecPath) > 0,
			`vectors.f32.bin should exist and be >0 (round=${round})`,
		);

		// 重新打开，验证“刚写入的 id”可被检索到
		const db2 = await reopen(storageDir);
		const results = await db2.search(v, { topK: 5 });

		assert(
			results.some((r: any) => r.id === id),
			`should find ${id} after reopen (round=${round})`,
		);

		await db2.close();
	}

	// best-effort 清理
	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ ATOMIC SAVE TEST PASSED");
}

main().catch((e) => {
	console.error("❌ ATOMIC SAVE TEST FAILED:", e);
	process.exit(1);
});
