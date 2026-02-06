import { MiniVectorDB } from "../src/index";

async function main() {
	console.log("--- BASIC TEST START ---");

	// Using default config (dim: 384)
	const db = new MiniVectorDB();
	await db.init();

	console.log("Inserting documents...");
	await db.insert({
		id: "doc-1",
		vector: new Array(384).fill(0.1),
		metadata: { title: "Hello Vector" },
	});

	await db.insert({
		id: "doc-2",
		vector: new Array(384).fill(0.2),
		metadata: { title: "World Vector" },
	});

	console.log("Searching for doc-1 pattern...");
	const query = new Array(384).fill(0.1);
	const results = await db.search(query, 1);

	console.log("Results:", JSON.stringify(results, null, 2));

	if (results.length > 0 && results[0].id === "doc-1") {
		console.log("✅ BASIC TEST PASSED");
	} else {
		console.error("❌ BASIC TEST FAILED");
		process.exit(1);
	}

	await db.close();
	process.exit(0);
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
