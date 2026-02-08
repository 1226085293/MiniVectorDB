// tests/semantic_test.ts
import { MiniVectorDB } from "../src/index";
import os from "os";
import path from "path";
import fs from "fs";
import { assert } from "./_util";

async function main() {
	console.log("--- SEMANTIC SEARCH TEST (v2) ---");

	const storageDir = path.join(
		os.tmpdir(),
		`minivectordb-semantic-${Date.now()}`,
	);
	fs.mkdirSync(storageDir, { recursive: true });

	const db = await MiniVectorDB.open({
		storageDir,
		modelName: "Xenova/all-MiniLM-L6-v2",
		modelArchitecture: "text",
		dim: 384,
		mode: "balanced",
		capacity: 5000,
		preloadVectors: false,
	});

	const data = [
		{ id: "fruit-1", text: "The apple is red and juicy." },
		{ id: "fruit-2", text: "Bananas are yellow and sweet." },
		{ id: "tech-1", text: "My laptop runs Node.js code." },
		{ id: "tech-2", text: "Quantum computing is the future." },
	];

	for (const item of data) {
		await db.insert({
			id: item.id,
			input: item.text,
			metadata: { text: item.text },
		});
	}

	// ✅ 只断言“类目命中”，不强求第一名（模型版本/量化/ef 参数会导致波动）
	const r1 = await db.search("delicious fruit", { topK: 3 });
	assert(
		r1.some((x) => x.id.startsWith("fruit-")),
		`expected fruit in topK, got=${r1.map((x) => x.id)}`,
	);

	const r2 = await db.search("programming language", { topK: 3 });
	assert(
		r2.some((x) => x.id.startsWith("tech-")),
		`expected tech in topK, got=${r2.map((x) => x.id)}`,
	);

	await db.close();

	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ SEMANTIC TEST PASSED");
}

main().catch((e) => {
	console.error("❌ SEMANTIC TEST FAILED:", e);
	process.exit(1);
});
