// tests/api_test.ts
import { spawn } from "child_process";
import path from "path";
import dotenv from "dotenv";
import net from "net";

// 加载本地 .env 配置
dotenv.config({ path: path.join(__dirname, "../.env") });

async function main() {
	console.log("--- API TEST START ---");

	const TEST_PORT = process.env.API_PORT || "3000";
	console.log(`Using port: ${TEST_PORT}`);
	console.log(`Using VECTOR_DIM: ${process.env.VECTOR_DIM || "384 (default)"}`);

	const serverPath = path.join(__dirname, "../src/api/server.ts");
	const cmd = process.platform === "win32" ? "npx.cmd" : "npx";

	const serverProcess = spawn(cmd, ["ts-node", serverPath], {
		stdio: ["ignore", "pipe", "pipe"],
		shell: true,
		env: {
			...process.env,
			PORT: TEST_PORT.toString(),
		},
	});

	let serverUrl = `http://localhost:${TEST_PORT}`;
	let outputBuffer = "";

	serverProcess.stdout.on("data", (data) => {
		const msg = data.toString();
		outputBuffer += msg;
		console.log("[SERVER OUT]", msg.trim());
	});

	serverProcess.stderr.on("data", (data) => {
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

		serverProcess.stdout.on("data", (data) => checkOutput(data.toString()));
		serverProcess.stderr.on("data", (data) => checkOutput(data.toString()));

		serverProcess.on("exit", (code) => {
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
		serverProcess.kill();
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
			const errData = await insertResp.json();
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
		process.exit(1);
	} finally {
		serverProcess.kill();
		if (process.platform === "win32" && serverProcess.pid) {
			try {
				spawn("taskkill", ["/pid", serverProcess.pid.toString(), "/f", "/t"]);
			} catch {}
		}
	}
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
