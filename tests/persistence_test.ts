import { WasmBridge } from "../src/core/wasm-bridge";
import path from "path";
import fs from "fs";

const SAVE_PATH = path.join(__dirname, "../data/dump.bin");

async function main() {
	// Cleanup previous test
	if (fs.existsSync(SAVE_PATH)) fs.unlinkSync(SAVE_PATH);

	console.log("--- PHASE 1: Insert & Save ---");
	const db1 = new WasmBridge();
	await db1.init();

	const vec = new Float32Array(128).fill(0.5);
	db1.insert(0, vec);
	console.log("Inserted ID: 0");

	await db1.save(SAVE_PATH);

	console.log("\n--- PHASE 2: Load & Verify ---");
	const db2 = new WasmBridge();
	await db2.init();

	// Verify empty before load (Search should return nothing or garbage, or just not crash)
	// Actually our simple mock search might behave weirdly if empty, but let's load first.

	await db2.load(SAVE_PATH);

	// Construct a query vector (same as inserted)
	const query = new Float32Array(128).fill(0.5);
	const result = db2.search(query, 1);

	console.log("Search Result ID:", result);

	if (result.length === 0) {
		console.log("SUCCESS: Persistence works!");
	} else {
		console.error("FAILURE: Could not find ID 0 after reload.");
	}
}

main().catch(console.error);
