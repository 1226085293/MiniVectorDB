import { MiniVectorDB } from "../src/index";
import path from "path";
import fs from "fs";
import os from "os";

async function main() {
	console.log("--- PERSISTENCE TEST START (v2 API) ---");

	// 用临时目录，避免污染项目 data/
	const storageDir = path.join(
		os.tmpdir(),
		`minivectordb-persist-${Date.now()}`,
	);
	if (!fs.existsSync(storageDir)) fs.mkdirSync(storageDir, { recursive: true });

	const dumpPath = path.join(storageDir, "dump.bin");
	const metaPath = path.join(storageDir, "metadata.json");
	const vecPath = path.join(storageDir, "vectors.f32.bin");

	// Clean up (如果目录复用时)
	for (const p of [dumpPath, metaPath, vecPath]) {
		if (fs.existsSync(p)) fs.unlinkSync(p);
	}

	// 1) Save data
	console.log("Stage 1: Inserting and Saving...");
	const db1 = await MiniVectorDB.open({
		storageDir,
		// 默认模型 + dim=384 就够；如果你想更确定可显式写 dim
		dim: 384,
		mode: "balanced",
		capacity: 10000,
		preloadVectors: false,
	});

	console.log("[stats]", db1.getStats());

	// 使用“唯一”的向量，降低碰撞/平局概率
	const vec = new Array(384).fill(0).map((_, i) => i / 384);

	await db1.insert({
		id: "persist-1",
		input: vec, // v2: input
		metadata: { info: "saved" },
	});

	// v2 save 默认写 storageDir/dump.bin（你也可传路径）
	await db1.save();
	await db1.close();
	console.log("Data saved and DB closed.");

	// 2) Load data
	console.log("\nStage 2: Loading and Verifying...");
	const db2 = await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity: 10000,
		preloadVectors: false,
	});

	console.log("[stats after reopen]", db2.getStats());

	// 重要：open 内部已经 try load()，这里可选再显式 load() 一次（幂等）
	await db2.load();

	console.log(`MetaDB has ${db2.getStats().items} items.`);
	console.log(
		`Files exist? dump=${fs.existsSync(dumpPath)} meta=${fs.existsSync(
			metaPath,
		)} vec=${fs.existsSync(vecPath)}`,
	);

	const results = await db2.search(vec, { topK: 1 }); // v2: search(query, {topK})
	console.log("Loaded Search Results:", JSON.stringify(results, null, 2));

	if (results.length > 0 && results[0].id === "persist-1") {
		console.log("✅ PERSISTENCE TEST PASSED");
	} else {
		console.error("❌ PERSISTENCE TEST FAILED: Result mismatch or empty");
		process.exit(1);
	}

	await db2.close();
	console.log("--- PERSISTENCE TEST END ---");
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
