import { MiniVectorDB } from "../src/index";
import path from "path";
import fs from "fs";

const DUMP_PATH = path.join(process.cwd(), "data/persistence_dump");
const META_PATH = path.join(process.cwd(), "data/persistence_meta.json");

async function main() {
	console.log("--- PERSISTENCE TEST START ---");

	// Clean up
	if (fs.existsSync(DUMP_PATH)) fs.unlinkSync(DUMP_PATH);
	if (fs.existsSync(META_PATH)) fs.unlinkSync(META_PATH);

	// 1. Save data
	console.log("Stage 1: Inserting and Saving...");
	const db1 = new MiniVectorDB({ metaDbPath: META_PATH });
	await db1.init();

	// Use a unique vector to ensure search accuracy
	const vec = new Array(384).fill(0).map((_, i) => i / 384);
	await db1.insert({
		id: "persist-1",
		vector: vec,
		metadata: { info: "saved" },
	});

	await db1.save(DUMP_PATH);
	await db1.close();
	console.log("Data saved and DB closed.");

	// 2. Load data
	console.log("\nStage 2: Loading and Verifying...");
	const db2 = new MiniVectorDB({ metaDbPath: META_PATH });
	await db2.init();
	await db2.load(DUMP_PATH);

	// Important: check if meta load worked
	console.log(`MetaDB has ${db2.getStats().items} items.`);

	const results = await db2.search(vec, 1);
	console.log("Loaded Search Results:", JSON.stringify(results, null, 2));

	if (results.length > 0 && results[0].id === "persist-1") {
		console.log("✅ PERSISTENCE TEST PASSED");
	} else {
		console.error("❌ PERSISTENCE TEST FAILED: Result mismatch or empty");
		process.exit(1);
	}

	await db2.close();
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
