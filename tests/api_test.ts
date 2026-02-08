// tests/api_test.ts
import { spawn } from "child_process";
import path from "path";
import dotenv from "dotenv";
import net from "net";
import os from "os";
import fs from "fs";

dotenv.config({ path: path.join(__dirname, "../.env") });

function delay(ms: number) {
	return new Promise((r) => setTimeout(r, ms));
}

function getAvailablePort(): Promise<number> {
	return new Promise((resolve, reject) => {
		const server = net.createServer();
		server.listen(0, "127.0.0.1", () => {
			const port = (server.address() as net.AddressInfo).port;
			server.close(() => resolve(port));
		});
		server.on("error", reject);
	});
}

async function killProcessTree(pid: number) {
	if (!pid) return;

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

async function waitExitOrKill(proc: any, timeoutMs: number) {
	const start = Date.now();
	while (Date.now() - start < timeoutMs) {
		if (proc.exitCode !== null) return;
		await delay(50);
	}
	await killProcessTree(proc.pid ?? 0);
}

async function fetchWithTimeout(url: string, init: any = {}, timeoutMs = 8000) {
	const controller = new AbortController();
	const tid = setTimeout(() => controller.abort(), timeoutMs);
	try {
		return await fetch(url, { ...init, signal: controller.signal });
	} finally {
		clearTimeout(tid);
	}
}

async function readJsonOrText(resp: Response) {
	const text = await resp.text().catch(() => "");
	try {
		return { json: JSON.parse(text), text };
	} catch {
		return { json: null, text };
	}
}

async function main() {
	console.log("--- API TEST START (v2 API) ---");

	const TEST_PORT = await getAvailablePort();
	console.log(`Using port: ${TEST_PORT}`);

	const vectorDim = parseInt(process.env.VECTOR_DIM || "384", 10);
	console.log(`Using VECTOR_DIM: ${vectorDim}`);

	const storageDir = path.join(
		os.tmpdir(),
		`minivectordb-api-test-${Date.now()}`,
	);
	fs.mkdirSync(storageDir, { recursive: true });
	console.log(`Using storage dir: ${storageDir}`);

	const serverPath = path.join(__dirname, "../src/api/server.ts");
	const cmd = process.platform === "win32" ? "npx.cmd" : "npx";

	const serverProcess = spawn(cmd, ["ts-node", serverPath], {
		stdio: ["ignore", "pipe", "pipe"],
		shell: true,
		env: {
			...process.env,
			API_PORT: TEST_PORT.toString(),
			API_HOST: "127.0.0.1",
			NODE_ENV: "test",
			MINIVECTOR_STORAGE_DIR: storageDir,
			VECTOR_DIM: String(vectorDim),
			MODEL_NAME: process.env.MODEL_NAME || "Xenova/all-MiniLM-L6-v2",
		},
	});

	const serverUrl = `http://127.0.0.1:${TEST_PORT}`;
	let outputBuffer = "";

	serverProcess.stdout?.on("data", (data: Buffer) => {
		const msg = data.toString();
		outputBuffer += msg;
		console.log("[SERVER OUT]", msg.trim());
	});

	serverProcess.stderr?.on("data", (data: Buffer) => {
		const msg = data.toString();
		outputBuffer += msg;
		console.error("[SERVER ERR]", msg.trim());
	});

	const ready = await new Promise<boolean>((resolve) => {
		let isReady = false;

		const checkOutput = (data: string) => {
			if (data.includes("Server listening") && !isReady) {
				isReady = true;
				resolve(true);
			}
		};

		serverProcess.stdout?.on("data", (data: Buffer) =>
			checkOutput(data.toString()),
		);
		serverProcess.stderr?.on("data", (data: Buffer) =>
			checkOutput(data.toString()),
		);

		serverProcess.on("exit", (code: number) => {
			if (!isReady) {
				console.error(`Server process exited with code ${code}`);
				resolve(false);
			}
		});

		setTimeout(() => {
			if (!isReady) {
				console.error("Server start timeout (15s)");
				resolve(false);
			}
		}, 15000);
	});

	if (!ready) {
		console.error("\n--- Server Output ---");
		console.error(outputBuffer);
		console.error("---------------------\n");
		await killProcessTree(serverProcess.pid ?? 0);
		process.exit(1);
	}

	try {
		// 0) sanity /stats
		console.log("Checking /stats ...");
		const statsResp = await fetchWithTimeout(`${serverUrl}/stats`, {}, 8000);
		const statsPayload = await readJsonOrText(statsResp);
		if (statsResp.status !== 200) {
			console.error("Stats failed:", statsPayload.text);
			throw new Error("/stats failed");
		}

		// 1) insert
		console.log(`Testing /insert with ${vectorDim} dim numeric vector...`);
		const insertResp = await fetchWithTimeout(
			`${serverUrl}/insert`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					id: "api-test-1",
					input: new Array(vectorDim).fill(0.1),
					metadata: { type: "api-test", phase: "inserted" },
				}),
			},
			8000,
		);

		const insertPayload = await readJsonOrText(insertResp);
		if (insertResp.status !== 200) {
			console.error("Insert failed:", insertPayload.text);
			throw new Error("API Insert Failed");
		}

		// 2) search must find id
		console.log("Testing /search (no filter) ...");
		const searchResp1 = await fetchWithTimeout(
			`${serverUrl}/search`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					query: new Array(vectorDim).fill(0.1),
					topK: 3,
				}),
			},
			8000,
		);

		const searchPayload1 = await readJsonOrText(searchResp1);
		if (searchResp1.status !== 200) {
			console.error("Search failed:", searchPayload1.text);
			throw new Error("API Search Failed");
		}

		const s1 = (searchPayload1.json ?? {}) as any;
		const results1 = s1?.results ?? s1; // 兼容：有的实现直接返回数组
		if (
			Array.isArray(results1) &&
			results1.some((x: any) => x?.id === "api-test-1")
		) {
			console.log("✅ Search returned inserted id.");
		} else {
			console.error("Search results payload:", searchPayload1.text);
			throw new Error("API Search mismatch (did not find api-test-1)");
		}

		// 3) updateMetadata（如果你实现了该端点）
		console.log("Testing /updateMetadata ...");
		const updResp = await fetchWithTimeout(
			`${serverUrl}/updateMetadata`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					id: "api-test-1",
					metadata: { type: "api-test", phase: "updated" },
				}),
			},
			8000,
		);

		const updPayload = await readJsonOrText(updResp);
		if (updResp.status !== 200) {
			console.error("updateMetadata failed:", updPayload.text);
			throw new Error("API updateMetadata failed");
		}

		// 4) search again (this is where your log previously “hangs”)
		console.log("Testing /search after updateMetadata (with timeout) ...");
		const searchResp2 = await fetchWithTimeout(
			`${serverUrl}/search`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					query: new Array(vectorDim).fill(0.1),
					topK: 3,
					// 如果你实现 filter，这里可以逐步加断言；先不强绑定 filter 语法
					// filter: { type: "api-test" },
				}),
			},
			8000,
		);

		const searchPayload2 = await readJsonOrText(searchResp2);
		if (searchResp2.status !== 200) {
			console.error("Search2 failed:", searchPayload2.text);
			throw new Error("API Search2 Failed");
		}

		const s2 = (searchPayload2.json ?? {}) as any;
		const results2 = s2?.results ?? s2;
		if (
			Array.isArray(results2) &&
			results2.some((x: any) => x?.id === "api-test-1")
		) {
			console.log("✅ Search after updateMetadata returned inserted id.");
		} else {
			console.error("Search2 results payload:", searchPayload2.text);
			throw new Error("API Search2 mismatch (did not find api-test-1)");
		}

		console.log("✅ API TEST PASSED!");
	} catch (e: any) {
		console.error("❌ API TEST FAILED:", e?.stack || e);
		process.exitCode = 1;
	} finally {
		// shutdown best-effort
		try {
			await fetchWithTimeout(`${serverUrl}/shutdown`, { method: "POST" }, 3000);
		} catch {}

		await waitExitOrKill(serverProcess, 8000);

		try {
			fs.rmSync(storageDir, { recursive: true, force: true });
		} catch {}

		process.exit(process.exitCode ?? 0);
	}
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
