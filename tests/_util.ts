// tests/_util.ts
import os from "os";
import path from "path";
import fs from "fs";
import net from "net";

export function delay(ms: number) {
	return new Promise((r) => setTimeout(r, ms));
}

export function mkTmpDir(prefix: string) {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
	return dir;
}

export function rmrf(p: string) {
	try {
		fs.rmSync(p, { recursive: true, force: true });
	} catch {}
}

export function assert(cond: any, msg: string) {
	if (!cond) throw new Error(msg);
}

export async function getAvailablePort(): Promise<number> {
	return new Promise((resolve, reject) => {
		const server = net.createServer();
		server.listen(0, "127.0.0.1", () => {
			const port = (server.address() as net.AddressInfo).port;
			server.close(() => resolve(port));
		});
		server.on("error", reject);
	});
}

export async function fetchJson(
	url: string,
	opts: RequestInit & { timeoutMs?: number } = {},
): Promise<{ status: number; json: any; text: string }> {
	const { timeoutMs = 8000, ...rest } = opts;

	const ac = new AbortController();
	const t = setTimeout(() => ac.abort(), timeoutMs);

	let status = 0;
	let text = "";
	let json: any = null;

	try {
		const resp = await fetch(url, { ...rest, signal: ac.signal });
		status = resp.status;
		text = await resp.text();
		try {
			json = text ? JSON.parse(text) : null;
		} catch {
			json = null;
		}
	} finally {
		clearTimeout(t);
	}

	return { status, json, text };
}

export async function waitForHttpReady(
	baseUrl: string,
	pathname = "/stats",
	timeoutMs = 15000,
) {
	const start = Date.now();
	let lastErr = "";

	while (Date.now() - start < timeoutMs) {
		try {
			const r = await fetchJson(`${baseUrl}${pathname}`, { timeoutMs: 2000 });
			if (r.status === 200) return r.json;
			lastErr = `status=${r.status} body=${r.text?.slice(0, 200)}`;
		} catch (e: any) {
			lastErr = String(e?.message || e);
		}
		await delay(200);
	}

	throw new Error(`Server not ready in ${timeoutMs}ms. lastErr=${lastErr}`);
}

// Windows 下必须杀进程树（npx/ts-node 常有子进程）
export async function killProcessTree(pid: number) {
	if (!pid) return;

	const { spawn } = await import("child_process");

	if (process.platform === "win32") {
		spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
			stdio: "ignore",
			shell: true,
		});
		return;
	}

	try {
		process.kill(pid, "SIGTERM");
	} catch {}
	await delay(200);
	try {
		process.kill(pid, "SIGKILL");
	} catch {}
}

export async function waitExitOrKill(proc: any, timeoutMs: number) {
	const start = Date.now();
	while (Date.now() - start < timeoutMs) {
		if (proc.exitCode !== null) return;
		await delay(50);
	}
	await killProcessTree(proc.pid ?? 0);
}
