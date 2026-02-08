// tests/capacity_overflow_test.ts
import { MiniVectorDB } from "../src/index";
import path from "path";
import os from "os";
import fs from "fs";
import { assert } from "./_util";

async function main() {
	console.log("--- CAPACITY OVERFLOW TEST START (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-capacity-"),
	);

	const capacity = 20;
	const db = await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity,
		preloadVectors: false,
	});

	const dim = db.getStats().dim;

	console.log(`Inserting ${capacity} docs (should succeed)...`);
	for (let i = 0; i < capacity; i++) {
		const v = new Array(dim).fill(0);
		v[i % dim] = 1;
		await db.insert({ id: `ok-${i}`, input: v, metadata: { i } });
	}

	console.log("Inserting one more doc (should fail)...");
	let thrown = false;
	let msg = "";

	try {
		const v = new Array(dim).fill(0);
		v[0] = 1;
		await db.insert({ id: "overflow-1", input: v, metadata: {} });
	} catch (e: any) {
		thrown = true;
		msg = String(e?.message || e);
	}

	assert(thrown, "expected insert to throw when capacity exceeded");

	// 兼容不同实现，但至少要包含一些关键提示词（越多越好）
	const lower = msg.toLowerCase();
	assert(
		lower.includes("capacity") ||
			lower.includes("full") ||
			lower.includes("rebuild"),
		`error message should mention capacity/rebuild guidance, got: ${msg}`,
	);

	await db.close();
	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("✅ CAPACITY OVERFLOW TEST PASSED");
}

main().catch((e) => {
	console.error("❌ CAPACITY OVERFLOW TEST FAILED:", e);
	process.exit(1);
});
