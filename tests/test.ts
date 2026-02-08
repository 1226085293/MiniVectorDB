// tests/test.ts
import { spawnSync } from "child_process";
import path from "path";

const tests = [
	"basic_test.ts",
	"persistence_test.ts",
	"semantic_test.ts",
	"image_test.ts",
	"search_many_test.ts",
	"rebuild_threshold_test.ts",
	"capacity_overflow_test.ts",
	"config_mismatch_test.ts",
	"deadlock_search_test.ts",
	"atomic_save_test.ts",
	"concurrent_open_close_test.ts",
	"api_test.ts",
];

function nowMs() {
	return Date.now();
}

async function main() {
	console.log("========================================");
	console.log("      MINIVECTORDB ALL TESTS RUNNER     ");
	console.log("========================================\n");

	let passed = 0;
	let failed = 0;

	for (const testFile of tests) {
		const start = nowMs();
		console.log(`[RUNNING]: ${testFile}`);

		const testPath = path.join(__dirname, testFile);
		const isWin = process.platform === "win32";
		const npx = isWin ? "npx.cmd" : "npx";

		const result = spawnSync(npx, ["ts-node", testPath], {
			stdio: ["ignore", "pipe", "pipe"],
			shell: true,
			encoding: "utf-8",
		});

		const cost = ((nowMs() - start) / 1000).toFixed(2);

		if (result.status === 0) {
			console.log(`[PASSED]: ${testFile} (${cost}s)\n`);
			passed++;
		} else {
			const combined = (result.stderr || result.stdout || "").toString().trim();
			const tail =
				combined.split(/\r?\n/).slice(-14).join("\n") || "(no output)";
			console.error(
				`[FAILED]: ${testFile} (Exit code: ${result.status}, ${cost}s)`,
			);
			console.error(`--- output (tail) ---\n${tail}\n`);
			failed++;
		}
	}

	console.log("========================================");
	console.log(`SUMMARY: ${passed} Passed, ${failed} Failed`);
	console.log("========================================");

	process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
	console.error(e);
	process.exit(1);
});
