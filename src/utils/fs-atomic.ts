// src/utils/fs-atomic.ts
import fs from "fs";

export async function atomicReplace(tmpPath: string, finalPath: string) {
	try {
		await fs.promises.rename(tmpPath, finalPath);
	} catch (e: any) {
		if (
			e?.code === "EEXIST" ||
			e?.code === "EPERM" ||
			e?.code === "ENOTEMPTY"
		) {
			await fs.promises.unlink(finalPath).catch(() => {});
			await fs.promises.rename(tmpPath, finalPath);
			return;
		}
		throw e;
	}
}

export function makeUniqueTmpPath(finalPath: string) {
	return (
		`${finalPath}.tmp-` +
		`${process.pid}-` +
		`${Date.now()}-` +
		Math.random().toString(16).slice(2)
	);
}
