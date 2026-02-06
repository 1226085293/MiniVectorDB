// tests/test.ts
import { spawnSync } from "child_process";
import path from "path";

const tests = [
	"basic_test.ts",
	"persistence_test.ts",
	"semantic_test.ts",
	"image_test.ts",
	"api_test.ts",
];

async function main() {
	console.log("========================================");
	console.log("      MINIVECTORDB ALL TESTS RUNNER     ");
	console.log("========================================\n");

	let passed = 0;
	let failed = 0;

	for (const testFile of tests) {
		console.log(`[RUNNING]: ${testFile}`);

		const testPath = path.join(__dirname, testFile);
		const isWin = process.platform === "win32";
		const npx = isWin ? "npx.cmd" : "npx";

		// ✅ 屏蔽子脚本输出：不要 inherit，改为 pipe
		const result = spawnSync(npx, ["ts-node", testPath], {
			stdio: ["ignore", "pipe", "pipe"],
			shell: true,
			encoding: "utf-8",
		});

		if (result.status === 0) {
			console.log(`[PASSED]: ${testFile}\n`);
			passed++;
		} else {
			// ✅ 只在失败时，给一个简短错误摘要（不打印子脚本所有日志）
			const errSnippet =
				(result.stderr || result.stdout || "")
					.toString()
					.trim()
					.split(/\r?\n/)
					.slice(-10)
					.join("\n") || "(no output)";
			console.error(`[FAILED]: ${testFile} (Exit code: ${result.status})`);
			console.error(`--- error (tail) ---\n${errSnippet}\n`);
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
