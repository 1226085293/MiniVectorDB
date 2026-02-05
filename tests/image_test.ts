import { MiniVectorDB } from "../src/index";
import fs from "fs";
import path from "path";

async function main() {
	console.log("--- MULTI-MODAL (IMAGE) SEARCH TEST ---");

	/**
	 * CLIP 模型 (Xenova/clip-vit-base-patch32) 产生 512 维向量
	 */
	const db = new MiniVectorDB({
		dim: 512,
		modelName: "Xenova/clip-vit-base-patch32",
	});

	await db.init();

	const img1Path = path.join(__dirname, "images/1.png");
	const img2Path = path.join(__dirname, "images/2.png");

	if (!fs.existsSync(img1Path)) {
		console.error(`❌ 错误: 找不到测试图片 ${img1Path}`);
		return;
	}

	console.log("Inserting image 1 into DB (using file path)... ");
	await db.insert({
		id: "image-1",
		vector: img1Path,
		metadata: { filename: "1.png" },
	});

	console.log("Inserting image 2 into DB (using file path)... ");
	await db.insert({
		id: "image-2",
		vector: img2Path,
		metadata: { filename: "2.png" },
	});

	console.log("\n--- Test 1: Text-to-Image Search ---");
	const textQuery = "a photo";
	console.log(`Searching for: "${textQuery}"...`);
	const results1 = await db.search(textQuery, 2);
	console.log("Search Results:", JSON.stringify(results1, null, 2));

	console.log("\n--- Test 2: Image-to-Image Search ---");
	console.log("Searching using image-1 as query...");
	const results2 = await db.search(img1Path, 2);
	console.log("Search Results:", JSON.stringify(results2, null, 2));

	if (results2.length > 0 && results2[0].id === "image-1") {
		console.log("✅ Success: Image matching works correctly.");
	}

	await db.close();
	console.log("\nTest completed.");
}

main().catch(console.error);
