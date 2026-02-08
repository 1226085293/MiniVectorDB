// src/utils/atomic.ts
import fs from "fs/promises";
import path from "path";

function isRetryableWinError(e: any) {
	const code = String(e?.code || "");
	// Windows 常见：EPERM(占用)、EBUSY、ENOENT(时序)、EACCES
	return (
		code === "EPERM" ||
		code === "EBUSY" ||
		code === "ENOENT" ||
		code === "EACCES"
	);
}

async function delay(ms: number) {
	return new Promise((r) => setTimeout(r, ms));
}

/**
 * atomicWriteFile:
 * 1) write to unique tmp in same dir
 * 2) fsync tmp
 * 3) rename tmp -> target (atomic on same volume)
 * 4) best-effort: fsync directory on non-win
 */
export async function atomicWriteFile(
	targetPath: string,
	data: string | Buffer | Uint8Array,
	opts?: { retries?: number; retryDelayMs?: number },
) {
	const retries = opts?.retries ?? 12;
	const retryDelayMs = opts?.retryDelayMs ?? 20;

	const dir = path.dirname(targetPath);
	const base = path.basename(targetPath);

	// 每次写唯一 tmp，避免并发冲突
	const tmpPath = path.join(
		dir,
		`${base}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
	);

	let fh: any = null;

	try {
		fh = await fs.open(tmpPath, "w");
		await fh.writeFile(data);

		// 尽量落盘
		try {
			await fh.sync();
		} catch {
			// 某些 fs/环境 sync 可能失败，忽略（仍然可用 rename 原子性）
		}
	} finally {
		try {
			await fh?.close();
		} catch {}
	}

	// rename 可能在 win 上被杀毒/索引器短暂占用，做 retry
	for (let i = 0; i <= retries; i++) {
		try {
			await fs.rename(tmpPath, targetPath);
			break;
		} catch (e: any) {
			if (
				process.platform === "win32" &&
				isRetryableWinError(e) &&
				i < retries
			) {
				await delay(retryDelayMs);
				continue;
			}
			// 如果 rename 失败，尽量清理 tmp
			try {
				await fs.unlink(tmpPath);
			} catch {}
			throw e;
		}
	}

	// best-effort: fsync directory（Windows 没必要且常失败）
	if (process.platform !== "win32") {
		try {
			const dirHandle = await fs.open(dir, "r");
			try {
				await dirHandle.sync();
			} finally {
				await dirHandle.close();
			}
		} catch {}
	}
}
