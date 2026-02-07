// tests/api_test.ts
import { spawn } from "child_process";
import path from "path";
import dotenv from "dotenv";
import net from "net";
import os from "os";
import fs from "fs";

// 加载本地 .env 配置
dotenv.config({ path: path.join(__dirname, "../.env") });

function delay(ms: number) {
	return new Promise((r) => setTimeout(r, ms));
}

// 获取随机可用端口
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

// Windows 下必须杀进程树（npx/ts-node 经常还有子进程）
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

async function main() {
	console.log("--- API TEST START (v2 API) ---");

	const TEST_PORT = await getAvailablePort();
	console.log(`Using port: ${TEST_PORT}`);

	// ✅ 这个测试走“数值向量直传”，所以 dim 取 env 或默认 384
	const vectorDim = parseInt(process.env.VECTOR_DIM || "384", 10);
	console.log(`Using VECTOR_DIM: ${vectorDim}`);

	// ✅ 用临时目录，避免污染 data/
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

			// ✅ v2 server.ts 读取 MINIVECTOR_STORAGE_DIR
			MINIVECTOR_STORAGE_DIR: storageDir,

			// ✅ 强制维度匹配（MiniVectorDB.open 会读 VECTOR_DIM）
			VECTOR_DIM: String(vectorDim),

			// ✅ 避免用户本地 .env 把它切到 clip 模型导致 dim 推断冲突
			// 这里强制 text 模型（只影响“字符串输入”，我们测试数值向量不依赖它）
			MODEL_NAME: process.env.MODEL_NAME || "Xenova/all-MiniLM-L6-v2",
		},
	});

	const serverUrl = `http://localhost:${TEST_PORT}`;
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
		console.log(`Testing /insert with ${vectorDim} dim numeric vector...`);

		// ✅ v2: /insert body uses { id, input, metadata }
		const insertResp = await fetch(`${serverUrl}/insert`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				id: "api-test-1",
				input: new Array(vectorDim).fill(0.1),
				metadata: { type: "api-test" },
			}),
		});

		if (insertResp.status !== 200) {
			const errData = await insertResp.json().catch(() => ({}));
			console.error("Insert failed:", errData);
			throw new Error("API Insert Failed");
		}

		console.log("Testing /search ...");

		// ✅ v2: /search body uses { query, topK, filter }
		const searchResp = await fetch(`${serverUrl}/search`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				query: new Array(vectorDim).fill(0.1),
				topK: 1,
			}),
		});

		if (searchResp.status !== 200) {
			const errData = await searchResp.json().catch(() => ({}));
			console.error("Search failed:", errData);
			throw new Error("API Search Failed");
		}

		const searchData: any = await searchResp.json();

		if (searchData.results && searchData.results[0]?.id === "api-test-1") {
			console.log("✅ API TEST PASSED!");
		} else {
			console.error("Search results:", searchData);
			throw new Error("API Search mismatch");
		}

		// (可选) 测一下 /stats 不挂
		const statsResp = await fetch(`${serverUrl}/stats`);
		const stats = await statsResp.json().catch(() => ({}));
		console.log("[stats endpoint]", stats);
	} catch (e) {
		console.error("❌ API TEST FAILED:", e);
		process.exitCode = 1;
	} finally {
		// ✅ 先请求 shutdown（忽略错误）
		try {
			await fetch(`${serverUrl}/shutdown`, { method: "POST" });
		} catch {}

		// ✅ 限时等待 server 退出，超时就杀树（否则 test 永远挂着）
		await waitExitOrKill(serverProcess, 8000);

		// ✅ best-effort 清理临时目录
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
