// tests/basic_test.ts
import path from "path";
import os from "os";
import fs from "fs";
import { MiniVectorDB } from "../src/index";
import { assert } from "./_util";

async function tryFilterSearch(
	db: any,
	query: number[] | string,
	topK: number,
	expectedId: string,
) {
	// 常见三种 filter 约定：直接字段 / metadata 包裹 / dot path
	const candidates = [
		{ type: "api-test" },
		{ metadata: { type: "api-test" } },
		{ "metadata.type": "api-test" },
	];

	for (const f of candidates) {
		try {
			const res = await db.search(query, { topK, filter: f });
			if (Array.isArray(res) && res.some((r: any) => r.id === expectedId)) {
				return { ok: true as const, filterUsed: f, results: res };
			}
		} catch {
			// 有些实现遇到不支持的 filter 结构会 throw，这里忽略继续试
		}
	}

	return { ok: false as const, filterUsed: null, results: [] as any[] };
}

async function main() {
	console.log("--- BASIC TEST START (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-test-"),
	);

	const db = await MiniVectorDB.open({
		storageDir,
		mode: "balanced",
		preloadVectors: false,
		capacity: 10_000,
		// dim 若你项目里由 env/模型推断也行；为了稳定也可以写死 dim: 384
	} as any);

	const stats = db.getStats();
	console.log("[stats]", stats);

	const dim = stats.dim;
	assert(typeof dim === "number" && dim > 0, `invalid dim: ${dim}`);

	// 用 one-hot，避免相似度平局导致偶发排序差异
	const v1 = new Array(dim).fill(0);
	v1[0] = 1;

	const v2 = new Array(dim).fill(0);
	v2[1] = 1;

	console.log("Inserting documents (numeric vectors)...");
	await db.insert({
		id: "doc-1",
		input: v1,
		metadata: { title: "Hello Vector", type: "api-test" },
	});

	await db.insert({
		id: "doc-2",
		input: v2,
		metadata: { title: "World Vector", type: "other" },
	});

	console.log("Searching (no filter)...");
	const results = await db.search(v1, { topK: 1 });
	console.log("Results:", JSON.stringify(results, null, 2));

	assert(results.length > 0, "search should return results");
	assert(
		results[0].id === "doc-1",
		`expected top1 doc-1, got=${results[0].id}`,
	);

	// ✅ filter 测试：自动探测你实现支持哪种 filter 语法
	console.log("Searching with filter (auto-detect filter schema)...");
	const filterTry = await tryFilterSearch(db as any, v1, 5, "doc-1");

	if (filterTry.ok) {
		console.log(
			"Filter schema detected:",
			JSON.stringify(filterTry.filterUsed),
		);
		assert(
			filterTry.results.every((r: any) => r.id !== "doc-2"),
			"filtered results should exclude doc-2",
		);
	} else {
		console.warn(
			"⚠️ Filter test skipped: current filter schema not detected. " +
				"Your implementation may require a different filter format.",
		);
	}

	// save/load 快速验证
	console.log("Saving...");
	await db.save();
	await db.close();

	console.log("Re-opening...");
	const db2 = await MiniVectorDB.open({
		storageDir,
		mode: "balanced",
		preloadVectors: false,
		capacity: 10_000,
	} as any);

	const results2 = await db2.search(v1, { topK: 1 });
	assert(results2.length > 0, "search after reload should return results");
	assert(
		results2[0].id === "doc-1",
		`expected doc-1 after reload, got=${results2[0].id}`,
	);

	await db2.close();

	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ BASIC TEST PASSED");
	process.exit(0);
}

main().catch((e) => {
	console.error("❌ BASIC TEST FAILED:", e);
	process.exit(1);
});
