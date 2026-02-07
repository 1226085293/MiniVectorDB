// MiniVectorDB/tests/basic_test.ts
import path from "path";
import os from "os";
import fs from "fs";
import { MiniVectorDB } from "../src/index";

async function main() {
	console.log("--- BASIC TEST START (v2 API) ---");

	// ✅ 使用临时目录，避免污染项目 ./data
	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-test-"),
	);

	// ✅ 推荐：用 open() 一行打开（会自动 init + 尝试 load）
	const db = await MiniVectorDB.open({
		storageDir,
		mode: "balanced",
		// 如果你不想触发 embedder，可不传 modelName（会用默认）
		// modelName: "Xenova/all-MiniLM-L6-v2",
		preloadVectors: false,
		// 小测试用小容量即可（默认 1_200_000 也行）
		capacity: 10_000,
	});

	console.log("[stats]", db.getStats());

	// ✅ 不再用 vector 字段：新版本 insert 使用 input
	console.log("Inserting documents (numeric vectors)...");
	const dim = db.getStats().dim;

	await db.insert({
		id: "doc-1",
		input: new Array(dim).fill(0.1),
		metadata: { title: "Hello Vector" },
	});

	await db.insert({
		id: "doc-2",
		input: new Array(dim).fill(0.2),
		metadata: { title: "World Vector" },
	});

	console.log("Searching for doc-1 pattern...");
	const query = new Array(dim).fill(0.1);

	// ✅ 新版 search(query, { topK, filter })
	const results = await db.search(query, { topK: 1 });

	console.log("Results:", JSON.stringify(results, null, 2));

	if (results.length > 0 && results[0].id === "doc-1") {
		console.log("✅ BASIC TEST PASSED");
	} else {
		console.error("❌ BASIC TEST FAILED");
		process.exitCode = 1;
	}

	// ✅ 可选：测试 save/load（验证持久化是否正常）
	console.log("Saving...");
	await db.save();

	console.log("Closing...");
	await db.close();

	console.log("Re-opening and loading...");
	const db2 = await MiniVectorDB.open({
		storageDir,
		mode: "balanced",
		preloadVectors: false,
		capacity: 10_000,
	});

	const results2 = await db2.search(query, { topK: 1 });
	console.log("Results after reload:", JSON.stringify(results2, null, 2));

	if (results2.length > 0 && results2[0].id === "doc-1") {
		console.log("✅ SAVE/LOAD TEST PASSED");
	} else {
		console.error("❌ SAVE/LOAD TEST FAILED");
		process.exitCode = 1;
	}

	await db2.close();

	// ✅ 清理临时目录（失败时也建议保留排查；这里默认清理）
	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log("--- BASIC TEST END ---");

	// 让 CI 正确拿到 exitCode
	process.exit(process.exitCode ?? 0);
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
