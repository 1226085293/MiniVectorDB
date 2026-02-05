import { MiniVectorDB } from '../src/index';

async function main() {
    console.log("--- SEMANTIC SEARCH TEST ---");
    
    const db = new MiniVectorDB();
    await db.init();

    const data = [
        { id: "fruit-1", text: "The apple is red and juicy." },
        { id: "fruit-2", text: "Bananas are yellow." },
        { id: "tech-1", text: "My laptop runs Node.js code." },
        { id: "tech-2", text: "Quantum computing is the future." }
    ];

    console.log("Inserting documents (Auto-Embedding)...");
    for (const item of data) {
        process.stdout.write(`Embedding ${item.id}... `);
        await db.insert({
            id: item.id,
            vector: item.text,
            metadata: { text: item.text }
        });
        console.log("Done.");
    }

    console.log("\nQuery: 'delicious fruit'");
    const results1 = await db.search("delicious fruit", 2);
    console.log("Results:", results1.map(r => r.id));

    if (results1[0].id.includes("fruit")) {
        console.log("✅ Logic Check Passed: Fruit found first.");
    } else {
        console.error("❌ Logic Check Failed: Expected fruit.");
    }

    console.log("\nQuery: 'programming language'");
    const results2 = await db.search("programming language", 2);
    console.log("Results:", results2.map(r => r.id));

    if (results2[0].id.includes("tech")) {
        console.log("✅ Logic Check Passed: Tech found first.");
    } else {
        console.error("❌ Logic Check Failed: Expected tech.");
    }

    await db.close();
}

main().catch(console.error);