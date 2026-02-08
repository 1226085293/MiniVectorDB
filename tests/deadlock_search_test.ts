// tests/deadlock_search_test.ts
import { MiniVectorDB } from "../src/index";
import os from "os";
import path from "path";
import fs from "fs";
import { assert } from "./_util";

function delay(ms: number) {
	return new Promise((r) => setTimeout(r, ms));
}

async function withTimeout<T>(
	p: Promise<T>,
	ms: number,
	label: string,
): Promise<T> {
	let t: any;
	const timeout = new Promise<T>((_, reject) => {
		t = setTimeout(() => reject(new Error(`Timeout(${ms}ms): ${label}`)), ms);
	});
	try {
		return await Promise.race([p, timeout]);
	} finally {
		clearTimeout(t);
	}
}

async function main() {
	console.log("--- DEADLOCK SEARCH TEST START (v2) ---");

	const storageDir = fs.mkdtempSync(
		path.join(os.tmpdir(), "minivectordb-deadlock-"),
	);

	const db = await MiniVectorDB.open({
		storageDir,
		dim: 384,
		mode: "balanced",
		capacity: 50_000,
		preloadVectors: false,
	});

	const dim = db.getStats().dim;
	assert(dim === 384, `expected dim=384, got=${dim}`);

	// 先塞一点数据，避免空库搜索异常分支
	for (let i = 0; i < 50; i++) {
		await db.insert({
			id: `seed-${i}`,
			input: new Array(dim).fill(0).map((_, k) => (k + i) / (dim + 1)),
			metadata: { kind: "seed" },
		});
	}

	const stopAt = Date.now() + 3500; // 测试强度窗口
	let inserts = 0;
	let searches = 0;
	let saves = 0;

	// 并发插入：持续制造写锁压力
	const insertWorker = async () => {
		while (Date.now() < stopAt) {
			const id = `ins-${inserts++}-${Date.now()}`;
			const v = new Array(dim).fill(0).map((_, k) => (k + inserts) / (dim + 7));
			await db.insert({ id, input: v, metadata: { kind: "ins" } });
			if (inserts % 20 === 0) await delay(10);
		}
	};

	// 并发搜索：持续制造读锁压力（search 内部可能会读 vectors/metadata）
	const searchWorker = async () => {
		while (Date.now() < stopAt) {
			const q = new Array(dim).fill(0.1);
			await db.search(q, { topK: 5 });
			searches++;
			if (searches % 40 === 0) await delay(5);
		}
	};

	// 保存压力：低频 save，测试“读写锁切换/锁升级”是否有死锁
	const saveWorker = async () => {
		while (Date.now() < stopAt) {
			await delay(120);
			await db.save();
			saves++;
		}
	};

	// 关键：整段并发必须在超时内结束，否则视为死锁/卡住
	await withTimeout(
		Promise.all([
			insertWorker(),
			searchWorker(),
			searchWorker(), // 两个搜索 worker 增加压力
			saveWorker(),
		]),
		12000,
		"deadlock scenario (insert/search/save)",
	);

	// 最后再做一次 save + search，确保状态正常
	await withTimeout(db.save(), 8000, "final save");
	const r = await withTimeout(
		db.search(new Array(dim).fill(0.1), { topK: 3 }),
		8000,
		"final search",
	);
	assert(Array.isArray(r), "final search should return array");

	await withTimeout(db.close(), 8000, "close");

	// best-effort 清理
	try {
		fs.rmSync(storageDir, { recursive: true, force: true });
	} catch {}

	console.log(
		`✅ DEADLOCK SEARCH TEST PASSED (inserts=${inserts}, searches=${searches}, saves=${saves})`,
	);
}

main().catch((e) => {
	console.error("❌ DEADLOCK SEARCH TEST FAILED:", e);
	process.exit(1);
});
