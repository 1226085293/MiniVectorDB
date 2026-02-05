import { spawn, ChildProcess } from 'child_process';
import path from 'path';

// Use global fetch (Node 18+) or install node-fetch if older
// @ts-ignore
const fetch = global.fetch;

async function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
    console.log("--- API TEST START ---");
    const serverPath = path.join(__dirname, '../src/api/server.ts');
    
    // Start Server as a child process
    console.log("Starting server...");
    // Use npx ts-node to run the server
    // On Windows, use 'npx.cmd'
    const isWin = process.platform === "win32";
    const cmd = isWin ? "npx.cmd" : "npx";
    
    const TEST_PORT = 3001;
    const serverProcess = spawn(cmd, ["ts-node", serverPath], {
        stdio: 'pipe',
        shell: true,
        env: { ...process.env, PORT: TEST_PORT.toString() }
    });

    let serverUrl = `http://localhost:${TEST_PORT}`;
    let ready = false;

    serverProcess.stdout.on('data', (data) => {
        const msg = data.toString();
        console.log("[SERVER]:", msg.trim());
        if (msg.includes("Server listening")) {
            ready = true;
        }
        // Also check for EADDRINUSE in stdout/stderr if printed there
        if (msg.includes("EADDRINUSE")) {
             console.error("Port " + TEST_PORT + " is already in use!");
             ready = false;
             process.exit(1);
        }
    });
    
    serverProcess.stderr.on('data', (data) => console.error("[SERVER ERR]:", data.toString()));

    // Wait for server to be ready
    let retries = 20;
    while (!ready && retries > 0) {
        await sleep(500);
        retries--;
    }
    
    if (!ready) {
        console.error("Server failed to start in time.");
        serverProcess.kill();
        process.exit(1);
    }

    try {
        // 1. Insert
        console.log("\n1. Testing /insert ...");
        const vec = new Array(128).fill(0.1);
        const insertRes = await fetch(`${serverUrl}/insert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                id: "api-test-1",
                vector: vec,
                metadata: { type: "test-api", lang: "js" }
            })
        });
        const insertJson = await insertRes.json();
        console.log("Insert Response:", insertJson);
        if (insertJson.status !== "ok") throw new Error("Insert failed");

        // 2. Search
        console.log("\n2. Testing /search ...");
        const searchRes = await fetch(`${serverUrl}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                vector: vec,
                k: 5,
                filter: { lang: "js" }
            })
        });
        const searchJson = await searchRes.json();
        console.log("Search Response:", JSON.stringify(searchJson, null, 2));
        
        if (!searchJson.results || searchJson.results.length === 0) {
            throw new Error("Search returned no results");
        }
        if (searchJson.results[0].id !== "api-test-1") {
            throw new Error("Search returned wrong ID");
        }

        console.log("\nAPI TEST PASSED!");
    } catch (e) {
        console.error("API TEST FAILED:", e);
    } finally {
        console.log("Stopping server...");
        // Kill process tree on Windows is tricky, but basic kill might work for testing
        // Using taskkill on windows to be sure if simple kill fails
        if (isWin) {
             spawn("taskkill", ["/pid", serverProcess.pid!.toString(), "/f", "/t"]);
        } else {
             serverProcess.kill();
        }
        process.exit(0);
    }
}

main();