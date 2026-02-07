// tests/multimodal_image_test.ts
import { MiniVectorDB } from "../src/index";
import fs from "fs";
import path from "path";
import os from "os";

async function main() {
	console.log("--- MULTI-MODAL (IMAGE) SEARCH TEST (v2 API) ---");

	// 你可以把测试数据写到 temp，避免污染项目 data/
	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-image-test-"),
	);

	/**
	 * CLIP 模型 (Xenova/clip-vit-base-patch32) 产生 512 维向量
	 * v2：建议显式写 modelArchitecture:"clip"（不依赖字符串推断更稳）
	 */
	const db = await MiniVectorDB.open({
		storageDir,
		modelName: "Xenova/clip-vit-base-patch32",
		modelArchitecture: "clip",
		dim: 512, // ✅ 强制 512，避免 .env 覆盖
		mode: "balanced",
		capacity: 1000,
		preloadVectors: false,
	});

	console.log("[stats]", db.getStats());

	const img1Path = path.join(__dirname, "images/1.png");
	const img2Path = path.join(__dirname, "images/2.png");

	if (!fs.existsSync(img1Path)) {
		console.error(`❌ 找不到测试图片: ${img1Path}`);
		process.exit(1);
	}
	if (!fs.existsSync(img2Path)) {
		console.error(`❌ 找不到测试图片: ${img2Path}`);
		process.exit(1);
	}

	console.log("Inserting image 1 into DB (using file path)...");
	await db.insert({
		id: "image-1",
		input: img1Path, // ✅ v2: input
		metadata: { filename: "1.png" },
	});

	console.log("Inserting image 2 into DB (using file path)...");
	await db.insert({
		id: "image-2",
		input: img2Path,
		metadata: { filename: "2.png" },
	});

	console.log("\n--- Test 1: Image-to-Image Search (deterministic) ---");
	console.log("Searching using image-1 as query...");
	const resultsImg = await db.search(img1Path, { topK: 2 });
	console.log("Search Results:", JSON.stringify(resultsImg, null, 2));

	if (!resultsImg.length || resultsImg[0].id !== "image-1") {
		console.error(
			"❌ FAILED: image-to-image search did not rank image-1 first.",
		);
		await db.close();
		process.exit(1);
	} else {
		console.log("✅ PASS: Image-to-image match ranks image-1 first.");
	}

	console.log("\n--- Test 2: Text-to-Image Search (demo, non-assert) ---");
	const textQuery = "a photo";
	console.log(`Searching for: "${textQuery}"...`);
	const resultsText = await db.search(textQuery, { topK: 2 });
	console.log("Search Results:", JSON.stringify(resultsText, null, 2));
	console.log(
		"ℹ️ Note: text->image ranking is content-dependent; not asserted.",
	);

	console.log("\n--- Test 3: Save/Load and repeat deterministic test ---");
	await db.save();
	await db.close();

	const db2 = await MiniVectorDB.open({
		storageDir,
		modelName: "Xenova/clip-vit-base-patch32",
		modelArchitecture: "clip",
		dim: 512, // ✅ 强制 512，避免 .env 覆盖
		mode: "balanced",
		capacity: 1000,
		preloadVectors: false,
	});

	const resultsAfter = await db2.search(img1Path, { topK: 2 });
	console.log("Results after reload:", JSON.stringify(resultsAfter, null, 2));

	if (!resultsAfter.length || resultsAfter[0].id !== "image-1") {
		console.error(
			"❌ FAILED: reload image-to-image search did not rank image-1 first.",
		);
		await db2.close();
		process.exit(1);
	} else {
		console.log("✅ PASS: Save/Load keeps deterministic ranking.");
	}

	await db2.close();
	console.log("\nTest completed.");
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
