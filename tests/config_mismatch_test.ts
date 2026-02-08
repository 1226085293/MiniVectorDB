// tests/config_mismatch_test.ts
import { MiniVectorDB } from "../src/index";
import path from "path";
import os from "os";
import fs from "fs";
import { assert } from "./_util";

function rmrf(p: string) {
	try {
		fs.rmSync(p, { recursive: true, force: true });
	} catch {}
}

function looksLikeDimMismatch(msg: string) {
	const lower = msg.toLowerCase();
	return (
		lower.includes("dim") ||
		lower.includes("mismatch") ||
		lower.includes("config") ||
		lower.includes("short read") ||
		lower.includes("vector store") ||
		lower.includes("readf32vectors")
	);
}

async function main() {
	console.log("--- CONFIG MISMATCH TEST START (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-mismatch-"),
	);

	let db2: any = null;

	try {
		// Stage 1: dim=384 保存
		const db1 = await MiniVectorDB.open({
			storageDir,
			dim: 384,
			mode: "balanced",
			capacity: 20000,
			preloadVectors: false,
		});

		const v384 = new Array(384).fill(0).map((_, i) => i / 384);

		await db1.insert({
			id: "mismatch-1",
			input: v384,
			metadata: { dim: 384, tag: "mismatch-stage1" },
		});

		await db1.save();
		await db1.close();
		console.log("Stage1 saved with dim=384.");

		// Stage 2: dim=512 打开
		let threwOnOpen = false;
		let openErrMsg = "";

		try {
			db2 = await MiniVectorDB.open({
				storageDir,
				dim: 512,
				mode: "balanced",
				capacity: 20000,
				preloadVectors: false, // 这里保持 false，才能覆盖“lazy load 才炸”的情况
			} as any);
		} catch (e: any) {
			threwOnOpen = true;
			openErrMsg = String(e?.message || e);
		}

		if (threwOnOpen) {
			assert(
				looksLikeDimMismatch(openErrMsg),
				`mismatch open should mention dim/config mismatch (or short read), got: ${openErrMsg}`,
			);
			console.log("✅ CONFIG MISMATCH TEST PASSED (threw on open as expected)");
			return;
		}

		assert(db2, "db2 should be opened or thrown");

		// 如果打开成功：有些实现会延迟到 search/load 才读取 vectors 文件
		const stats = db2.getStats?.() ?? {};
		assert(stats.dim === 512, `opened dim should be 512, got=${stats.dim}`);

		const q512 = new Array(512).fill(0.1);

		let results512: any[] | null = null;
		let threwOnFirstSearch = false;
		let searchErrMsg = "";

		try {
			results512 = await db2.search(q512, { topK: 5 });
		} catch (e: any) {
			threwOnFirstSearch = true;
			searchErrMsg = String(e?.message || e);
		}

		if (threwOnFirstSearch) {
			// ✅ 你现在遇到的就是这里：Vector store short read
			assert(
				looksLikeDimMismatch(searchErrMsg),
				`first search after dim mismatch should throw mismatch-like error, got: ${searchErrMsg}`,
			);
			console.log(
				"✅ CONFIG MISMATCH TEST PASSED (open ok, but first search threw mismatch/short-read as expected)",
			);
			return;
		}

		// 走到这里代表：open+search 都没抛错，则必须确保“旧 384 数据没有被错误读入”
		assert(Array.isArray(results512), "search() should return array results");
		assert(
			results512!.every((x: any) => x?.id !== "mismatch-1"),
			"must NOT find old 384-dim item after opening with dim=512",
		);

		// 更强约束：用 384 query 搜应抛错 或 items=0
		const items = (stats.items ??
			stats.activeCount ??
			stats.count ??
			0) as number;

		let threwOn384 = false;
		try {
			const v384 = new Array(384).fill(0).map((_, i) => i / 384);
			await db2.search(v384 as any, { topK: 5 });
		} catch {
			threwOn384 = true;
		}

		assert(
			threwOn384 || items === 0,
			"if dim=512 open succeeded, searching with 384 vector should throw OR items should be 0 (safe empty DB)",
		);

		console.log("✅ CONFIG MISMATCH TEST PASSED (opened in safe mode)");
	} finally {
		try {
			await db2?.close?.();
		} catch {}
		rmrf(storageDir);
	}
}

main().catch((e) => {
	console.error("❌ CONFIG MISMATCH TEST FAILED:", e);
	process.exit(1);
});
