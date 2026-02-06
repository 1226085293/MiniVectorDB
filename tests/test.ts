import { spawnSync } from 'child_process';
import path from 'path';

const tests = [
    'basic_test.ts',
    'persistence_test.ts',
    'semantic_test.ts',
    'image_test.ts',
    'api_test.ts'
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
        const isWin = process.platform === 'win32';
        const npx = isWin ? 'npx.cmd' : 'npx';

        const result = spawnSync(npx, ['ts-node', testPath], {
            stdio: 'inherit',
            shell: true
        });

        if (result.status === 0) {
            console.log(`\n[PASSED]: ${testFile}\n`);
            passed++;
        } else {
            console.error(`\n[FAILED]: ${testFile} (Exit code: ${result.status})\n`);
            failed++;
        }
    }

    console.log("========================================");
    console.log(`SUMMARY: ${passed} Passed, ${failed} Failed`);
    console.log("========================================");

    if (failed > 0) {
        process.exit(1);
    } else {
        process.exit(0);
    }
}

main();