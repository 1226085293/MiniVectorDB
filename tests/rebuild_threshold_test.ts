// tests/rebuild_threshold_test.ts
import { MiniVectorDB } from "../src/index";
import path from "path";
import os from "os";
import fs from "fs";
import { assert } from "./_util";

async function main() {
	console.log("--- REBUILD THRESHOLD TEST START (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-rebuild-thr-"),
	);

	// 注意：deletedRebuildThreshold 是否存在取决于你实现
	// 为了 TS 不报错，用 as any 传入可选参数
	const db = await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity: 20000,
		preloadVectors: false,
		// 如果你支持这个参数，会让测试更敏感
		// deletedRebuildThreshold: 0.2,
	} as any);

	const dim = db.getStats().dim;
	assert(dim === 384, `dim should be 384, got=${dim}`);

	// 插入 N 条：每条用唯一向量，避免平局
	const N = 200;
	console.log(`Inserting ${N} docs...`);
	for (let i = 0; i < N; i++) {
		const v = new Array(dim).fill(0);
		v[i % dim] = 1;
		await db.insert({
			id: `doc-${i}`,
			input: v,
			metadata: { i },
		});
	}

	// 删除 60%（如果你启用 threshold，这里应该触发 auto rebuild 或至少进入“需要重建”的状态）
	const delCount = Math.floor(N * 0.6);
	console.log(`Deleting ${delCount} docs...`);
	const idsToDelete = Array.from({ length: delCount }, (_, i) => `doc-${i}`);
	const rm = await db.removeMany(idsToDelete);
	// removeMany 返回值结构你实现可能不同，这里只做弱断言
	assert(rm && typeof rm === "object", "removeMany should return an object");

	// 关键断言：已删除的 doc 不应再被 search 返回（即使 ANN 里残留，也应在结果层过滤）
	console.log("Verifying deleted docs are excluded from search results...");
	for (let i = 0; i < 10; i++) {
		const idx = i; // 删除区间内
		const v = new Array(dim).fill(0);
		v[idx % dim] = 1;

		const results = await db.search(v, { topK: 10 });
		assert(
			results.every((r) => r.id !== `doc-${idx}`),
			`deleted doc-${idx} should not appear in search results: got=${results.map((x) => x.id).join(",")}`,
		);
	}

	// 如果你实现了 auto rebuild，这里可能已经发生；无论如何，我们再手动 rebuild 一次做“确定性收敛”
	console.log("Calling rebuild() to ensure index compactness...");
	await db.rebuild({ capacity: 25000, persist: false } as any);

	// rebuild 后，再验证一次：未删除的仍可命中
	console.log("Verifying remaining docs are still searchable after rebuild...");
	for (let i = delCount; i < delCount + 10; i++) {
		const v = new Array(dim).fill(0);
		v[i % dim] = 1;
		const results = await db.search(v, { topK: 3 });
		assert(results.length > 0, "search should return results");
		assert(
			results.some((r) => r.id === `doc-${i}`),
			`expected doc-${i} in results after rebuild`,
		);
	}

	await db.close();
	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ REBUILD THRESHOLD TEST PASSED");
}

main().catch((e) => {
	console.error("❌ REBUILD THRESHOLD TEST FAILED:", e);
	process.exit(1);
});
