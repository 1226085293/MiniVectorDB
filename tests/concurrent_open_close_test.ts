import { MiniVectorDB } from "../src/index";
import os from "os";
import path from "path";
import fs from "fs";

function delay(ms: number) {
	return new Promise((r) => setTimeout(r, ms));
}

async function main() {
	console.log("--- CONCURRENT OPEN/CLOSE TEST ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-concurrent-"),
	);

	const dim = 384;
	const rounds = 20;
	const workers = 6;

	const vec = new Array(dim).fill(0).map((_, i) => i / dim);

	const tasks = new Array(workers).fill(0).map((_, wi) =>
		(async () => {
			for (let r = 0; r < rounds; r++) {
				const db = await MiniVectorDB.open({
					storageDir,
					dim,
					mode: "balanced",
					capacity: 50_000,
					preloadVectors: false,
				});

				// 少量随机错峰，制造竞争
				await delay((Math.random() * 30) | 0);

				await db.insert({
					id: `w${wi}-r${r}`,
					input: vec,
					metadata: { wi, r },
				});

				// 再错峰
				await delay((Math.random() * 30) | 0);

				// save + search + close 都参与竞争
				await db.save();

				const res = await db.search(vec, { topK: 1 });
				if (!res.length) throw new Error("search returned empty after save");

				await db.close();
			}
		})(),
	);

	try {
		await Promise.all(tasks);
		console.log("✅ CONCURRENT OPEN/CLOSE TEST PASSED");
	} catch (e) {
		console.error("❌ CONCURRENT OPEN/CLOSE TEST FAILED:", e);
		process.exit(1);
	} finally {
		try {
			fs.rmSync(storageDir, { recursive: true, force: true });
		} catch {}
	}
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
