import { WasmBridge } from '../src/core/wasm-bridge';
import { MetaDB } from '../src/storage/meta-db';

async function main() {
    console.log("Starting basic test...");
    
    // 1. Test MetaDB (Pure JS)
    console.log("Initializing MetaDB...");
    const meta = new MetaDB();
    await meta.ready();
    meta.add("doc-1", 0, { title: "Hello World", tag: "test" });
    meta.add("doc-2", 1, { title: "Vector DB", tag: "prod" });
    
    const item = meta.get("doc-1");
    console.log("MetaDB fetch:", item?.metadata);

    // 2. Test WASM
    const db = new WasmBridge();
    await db.init();

    // Create a dummy vector (DIM=128)
    const vec1 = new Float32Array(128).fill(0.1);
    const vec2 = new Float32Array(128).fill(0.2);

    console.log("Inserting vector 0...");
    db.insert(0, vec1);
    
    console.log("Inserting vector 1...");
    db.insert(1, vec2);

    console.log("Searching...");
    // This won't return real results yet as search is mocked
    const res = db.search(vec1, 1);
    console.log("Search Result Ptr:", res);
}

main().catch(console.error);
