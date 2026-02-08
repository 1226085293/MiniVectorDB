// tests/image_test.ts
import { MiniVectorDB } from "../src/index";
import fs from "fs";
import path from "path";
import os from "os";
import { assert } from "./_util";

async function main() {
	console.log("--- MULTI-MODAL (IMAGE) SEARCH TEST (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-image-"),
	);

	const db = await MiniVectorDB.open({
		storageDir,
		modelName: "Xenova/clip-vit-base-patch32",
		modelArchitecture: "clip",
		dim: 512,
		mode: "balanced",
		capacity: 5000,
		preloadVectors: false,
	});

	const img1Path = path.join(__dirname, "images/1.png");
	const img2Path = path.join(__dirname, "images/2.png");

	assert(fs.existsSync(img1Path), `missing image: ${img1Path}`);
	assert(fs.existsSync(img2Path), `missing image: ${img2Path}`);

	await db.insert({
		id: "image-1",
		input: img1Path,
		metadata: { filename: "1.png" },
	});
	await db.insert({
		id: "image-2",
		input: img2Path,
		metadata: { filename: "2.png" },
	});

	const results = await db.search(img1Path, { topK: 2 });
	assert(results.length > 0, "image search should return results");
	assert(
		results[0].id === "image-1",
		`expected image-1 rank1, got=${JSON.stringify(results)}`,
	);

	await db.save();
	await db.close();

	const db2 = await MiniVectorDB.open({
		storageDir,
		modelName: "Xenova/clip-vit-base-patch32",
		modelArchitecture: "clip",
		dim: 512,
		mode: "balanced",
		capacity: 5000,
		preloadVectors: false,
	});

	const results2 = await db2.search(img1Path, { topK: 2 });
	assert(
		results2.length > 0 && results2[0].id === "image-1",
		"save/load should preserve deterministic hit",
	);

	await db2.close();

	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ IMAGE TEST PASSED");
}

main().catch((e) => {
	console.error("❌ IMAGE TEST FAILED:", e);
	process.exit(1);
});
