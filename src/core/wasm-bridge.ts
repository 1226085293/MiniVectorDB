import fs from "fs";
import loader from "@assemblyscript/loader";
import path from "path";

const WASM_PATH = path.join(__dirname, "../../build/release.wasm");

export class WasmBridge {
	private wasmModule: any;
	private memory!: WebAssembly.Memory;
	private instance!: WebAssembly.Instance;

	private currentConfig: { dim: number; m: number; ef: number } | null = null;

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

		// Initialize allocator
		this.wasmModule.init_memory();

		// Apply config (must be before init_index / load_index if your AS checks config)
		if (config) {
			this.currentConfig = config;
			this.wasmModule.set_config(config.dim, config.m, config.ef);
		}

		// Initialize an empty index by default (load_index will re-init internally)
		this.wasmModule.init_index(10000);

		console.log(
			"WASM Initialized. Memory size:",
			this.memory.buffer.byteLength,
		);
	}

	// Write Float32 vector into WASM linear memory and return ptr
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

		// Each result: [i32 id, f32 dist] => 8 bytes
		const view = new DataView(this.memory.buffer, resultsPtr, count * 8);

		for (let i = 0; i < count; i++) {
			const id = view.getInt32(i * 8, true);
			const dist = view.getFloat32(i * 8 + 4, true);
			results.push({ id, dist });
		}

		return results;
	}

	// Correct save: allocate exact dump size, save_index returns used bytes
	async save(filePath: string) {
		if (typeof this.wasmModule.get_index_dump_size !== "function") {
			throw new Error(
				"WASM export get_index_dump_size() not found. Did you update assembly/hnsw.ts?",
			);
		}

		const dumpSize: number = this.wasmModule.get_index_dump_size();
		const ptr: number = this.wasmModule.alloc(dumpSize);
		const used: number = this.wasmModule.save_index(ptr);

		const view = new Uint8Array(this.memory.buffer, ptr, used);
		await fs.promises.writeFile(filePath, view);

		console.log(`Index saved: ${used} bytes`);
	}

	// Correct load:
	// - reset_memory() to reclaim previous allocations (WASM cannot shrink; this is the clean way)
	// - re-apply config (if your loader validates dump config against current config)
	// - load_index() will re-init_index internally and rebuild pointers
	async load(filePath: string) {
		if (!fs.existsSync(filePath)) return;

		const buf = await fs.promises.readFile(filePath);

		// Reclaim old allocations to avoid growth across repeated loads
		if (typeof this.wasmModule.reset_memory === "function") {
			this.wasmModule.reset_memory();
		} else {
			// Fallback: at least re-init allocator base
			this.wasmModule.init_memory();
		}

		// Ensure config is applied before load (important if AS validates dump config)
		if (this.currentConfig) {
			this.wasmModule.set_config(
				this.currentConfig.dim,
				this.currentConfig.m,
				this.currentConfig.ef,
			);
		}

		// Copy dump into WASM memory
		const ptr = this.wasmModule.alloc(buf.length);
		new Uint8Array(this.memory.buffer, ptr, buf.length).set(buf);

		// Load dump
		this.wasmModule.load_index(ptr, buf.length);

		console.log(`Index loaded: ${buf.length} bytes`);
	}
}
