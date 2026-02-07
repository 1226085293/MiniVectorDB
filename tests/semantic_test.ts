import { MiniVectorDB } from "../src/index";
import os from "os";
import path from "path";
import fs from "fs";

async function main() {
	console.log("--- SEMANTIC SEARCH TEST (v2 API) ---");

	// 用临时目录避免污染
	const storageDir = path.join(
		os.tmpdir(),
		`minivectordb-semantic-${Date.now()}`,
	);
	fs.mkdirSync(storageDir, { recursive: true });

	const db = await MiniVectorDB.open({
		storageDir,
		// 默认 text 模型 Xenova/all-MiniLM-L6-v2 => 384
		modelName: "Xenova/all-MiniLM-L6-v2",
		mode: "balanced",
		capacity: 1000,
		preloadVectors: false,
	});

	console.log("[stats]", db.getStats());

	const data = [
		{ id: "fruit-1", text: "The apple is red and juicy." },
		{ id: "fruit-2", text: "Bananas are yellow." },
		{ id: "tech-1", text: "My laptop runs Node.js code." },
		{ id: "tech-2", text: "Quantum computing is the future." },
	];

	console.log("Inserting documents (Auto-Embedding)...");
	for (const item of data) {
		process.stdout.write(`Embedding ${item.id}... `);
		await db.insert({
			id: item.id,
			input: item.text, // v2: input 支持 string（自动 embed）
			metadata: { text: item.text },
		});
		console.log("Done.");
	}

	// --- Query 1 ---
	console.log("\nQuery: 'delicious fruit'");
	const results1 = await db.search("delicious fruit", { topK: 2 }); // v2: { topK }
	console.log(
		"Results:",
		results1.map((r) => r.id),
	);

	const hasFruit = results1.some((r) => r.id.startsWith("fruit-"));
	if (results1[0]?.id.startsWith("fruit-")) {
		console.log("✅ Logic Check Passed: Fruit ranked first.");
	} else if (hasFruit) {
		console.log(
			"✅ Logic Check Passed: Fruit appears in topK (ranking may vary).",
		);
	} else {
		console.error("❌ Logic Check Failed: Expected fruit in topK.");
		process.exit(1);
	}

	// --- Query 2 ---
	console.log("\nQuery: 'programming language'");
	const results2 = await db.search("programming language", { topK: 2 });
	console.log(
		"Results:",
		results2.map((r) => r.id),
	);

	const hasTech = results2.some((r) => r.id.startsWith("tech-"));
	if (results2[0]?.id.startsWith("tech-")) {
		console.log("✅ Logic Check Passed: Tech ranked first.");
	} else if (hasTech) {
		console.log(
			"✅ Logic Check Passed: Tech appears in topK (ranking may vary).",
		);
	} else {
		console.error("❌ Logic Check Failed: Expected tech in topK.");
		process.exit(1);
	}

	await db.close();
	console.log("\n--- SEMANTIC SEARCH TEST END ---");
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
