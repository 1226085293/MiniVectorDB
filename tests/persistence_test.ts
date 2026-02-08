// tests/persistence_test.ts
import { MiniVectorDB } from "../src/index";
import path from "path";
import os from "os";
import fs from "fs";
import { assert } from "./_util";

async function main() {
	console.log("--- PERSISTENCE + OPLOG REPLAY TEST START (v2) ---");

	const storageDir = path.join(
		os.tmpdir(),
		`minivectordb-persist-${Date.now()}`,
	);
	fs.mkdirSync(storageDir, { recursive: true });

	// Stage 1: insert + save snapshot
	const db1 = await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity: 20000,
		preloadVectors: false,
	});

	const vecA = new Array(384).fill(0).map((_, i) => i / 384);
	await db1.insert({ id: "persist-A", input: vecA, metadata: { stage: 1 } });

	await db1.save(); // ✅ snapshot
	console.log("Stage1 saved.");

	// Stage 2: insert WITHOUT save (should be recovered via oplog replay)
	const vecB = new Array(384).fill(0).map((_, i) => (384 - i) / 384);
	await db1.insert({ id: "persist-B", input: vecB, metadata: { stage: 2 } });

	// ⚠️ 不调用 save，直接 close -> 模拟 crash 前只写 oplog
	await db1.close();
	console.log("Stage2 inserted without save; closed.");

	// Stage 3: reopen -> open() 内部 try load() + replay oplog
	const db2 = await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity: 20000,
		preloadVectors: false,
	});

	const rA = await db2.search(vecA, { topK: 1 });
	assert(
		rA.length > 0 && rA[0].id === "persist-A",
		"persist-A must be found after reopen",
	);

	const rB = await db2.search(vecB, { topK: 1 });
	assert(
		rB.length > 0 && rB[0].id === "persist-B",
		"persist-B must be found via oplog replay",
	);

	await db2.close();

	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ PERSISTENCE + OPLOG REPLAY TEST PASSED");
}

main().catch((e) => {
	console.error("❌ PERSISTENCE TEST FAILED:", e);
	process.exit(1);
});
