import fs from "fs";
import loader from "@assemblyscript/loader";
import path from "path";

const WASM_PATH = path.join(__dirname, "../../build/release.wasm");

export class WasmBridge {
	private wasmModule: any;
	private memory!: WebAssembly.Memory;
	private instance!: WebAssembly.Instance;

	constructor() {}

	async init(config?: { dim: number; m: number; ef: number }) {
		const wasmBuffer = fs.readFileSync(WASM_PATH);
		const imports = {
			env: {
				abort: (msg: number, file: number, line: number, col: number) => {
					console.error(`WASM Abort: ${line}:${col}`);
				},
				seed: () => Math.random(),
			},
		};

		const result = await loader.instantiate(wasmBuffer, imports);
		this.wasmModule = result.exports;
		this.instance = result.instance;
		this.memory = this.wasmModule.memory as WebAssembly.Memory;

		this.wasmModule.init_memory();

		if (config) {
			this.wasmModule.set_config(config.dim, config.m, config.ef);
		}

		this.wasmModule.init_index(10000);
		console.log(
			"WASM Initialized. Memory size:",
			this.memory.buffer.byteLength,
		);
	}

	// 写向量到 WASM
	writeVector(vector: Float32Array): number {
		const ptr = this.wasmModule.alloc(vector.length * 4);
		const view = new Float32Array(this.memory.buffer, ptr, vector.length);
		view.set(vector);
		return ptr;
	}

	insert(id: number, vector: Float32Array) {
		const ptr = this.writeVector(vector);
		this.wasmModule.insert(id, ptr);
	}

	search(vector: Float32Array, k: number) {
		const ptr = this.writeVector(vector);
		const count = this.wasmModule.search(ptr, k);
		if (count === 0) return [];

		const resultsPtr = this.wasmModule.get_results_ptr();
		const results: { id: number; dist: number }[] = [];
		const view = new DataView(this.memory.buffer, resultsPtr, count * 8);

		for (let i = 0; i < count; i++) {
			const id = view.getInt32(i * 8, true);
			const dist = view.getFloat32(i * 8 + 4, true);
			results.push({ id, dist });
		}

		return results;
	}

	// 统一 save/load，底层使用 index
	async save(filePath: string) {
		const bufSize = this.wasmModule.get_memory_usage();
		const ptr = this.wasmModule.alloc(bufSize);
		const used = this.wasmModule.save_index(ptr);
		const view = new Uint8Array(this.memory.buffer, ptr, used);
		await fs.promises.writeFile(filePath, view);
		console.log(`Index saved: ${used} bytes`);
	}

	async load(filePath: string) {
		if (!fs.existsSync(filePath)) return;

		const buf = await fs.promises.readFile(filePath);
		const ptr = this.wasmModule.alloc(buf.length);
		new Uint8Array(this.memory.buffer, ptr, buf.length).set(buf);
		this.wasmModule.load_index(ptr, buf.length);

		console.log(`Index loaded: ${buf.length} bytes`);
	}
}
