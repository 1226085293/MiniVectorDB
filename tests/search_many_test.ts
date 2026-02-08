// tests/search_many_test.ts
import { MiniVectorDB } from "../src/index";
import path from "path";
import os from "os";
import fs from "fs";
import { assert } from "./_util";

async function main() {
	console.log("--- SEARCHMANY TEST START (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-searchmany-"),
	);

	const db = await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity: 50000,
		preloadVectors: false,
	} as any);

	const dim = db.getStats().dim;

	// 插入 128 条：每条一个 one-hot 向量
	const N = 128;
	console.log(`Inserting ${N} docs...`);
	for (let i = 0; i < N; i++) {
		const v = new Array(dim).fill(0);
		v[i % dim] = 1;
		await db.insert({ id: `d-${i}`, input: v, metadata: { i } });
	}

	// 造 64 个 query（其中前 10 个严格可预测）
	const Q = 64;
	const queries: number[][] = [];
	const expectedTop1: string[] = [];

	for (let i = 0; i < Q; i++) {
		const idx = i; // one-hot 对齐 d-i
		const v = new Array(dim).fill(0);
		v[idx % dim] = 1;
		queries.push(v);
		expectedTop1.push(`d-${idx}`);
	}

	console.log(`Calling searchMany with ${Q} queries...`);
	// 兼容：如果你方法名/参数不同，用 as any
	const results = await (db as any).searchMany(queries, { topK: 3 });

	assert(Array.isArray(results), "searchMany should return array");
	assert(
		results.length === Q,
		`searchMany length mismatch expected=${Q} got=${results.length}`,
	);

	// 前 10 个做强断言 top1
	for (let i = 0; i < 10; i++) {
		const row = results[i];
		assert(
			Array.isArray(row) && row.length > 0,
			`row[${i}] should be non-empty`,
		);
		assert(
			row[0].id === expectedTop1[i],
			`expected top1 ${expectedTop1[i]} got=${row[0].id}`,
		);
	}

	await db.close();
	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ SEARCHMANY TEST PASSED");
}

main().catch((e) => {
	console.error("❌ SEARCHMANY TEST FAILED:", e);
	process.exit(1);
});
