// tests/api_test.ts
import { spawn } from "child_process";
import path from "path";
import dotenv from "dotenv";
import net from "net";

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
	console.log("--- API TEST START ---");

	const TEST_PORT = await getAvailablePort();
	console.log(`Using port: ${TEST_PORT}`);
	console.log(`Using VECTOR_DIM: ${process.env.VECTOR_DIM || "384 (default)"}`);

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

	const vectorDim = parseInt(process.env.VECTOR_DIM || "384", 10);

	try {
		console.log(`Testing /insert with ${vectorDim} dim vector...`);
		const insertResp = await fetch(`${serverUrl}/insert`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				id: "api-test-1",
				vector: new Array(vectorDim).fill(0.1),
				metadata: { type: "api-test" },
			}),
		});

		if (insertResp.status !== 200) {
			const errData = await insertResp.json().catch(() => ({}));
			console.error("Insert failed:", errData);
			throw new Error("API Insert Failed");
		}

		console.log("Testing /search ...");
		const searchResp = await fetch(`${serverUrl}/search`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ vector: new Array(vectorDim).fill(0.1), k: 1 }),
		});

		const searchData: any = await searchResp.json();

		if (searchData.results && searchData.results[0]?.id === "api-test-1") {
			console.log("✅ API TEST PASSED!");
		} else {
			console.error("Search results:", searchData);
			throw new Error("API Search mismatch");
		}
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

		process.exit(process.exitCode ?? 0);
	}
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
