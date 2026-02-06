import fs from "fs";
import loader from "@assemblyscript/loader";
import path from "path";

const WASM_PATH = path.join(__dirname, "../../build/release.wasm");

export class WasmBridge {
	private wasmModule: any;
	private memory!: WebAssembly.Memory;
	private instance!: WebAssembly.Instance;

	// Pointers/Offsets
	private inputBufferPtr: number = 0;

	constructor() {
		// We will use the memory exported by WASM
	}

	async init(config?: { dim: number; m: number; ef: number }) {
		const wasmBuffer = fs.readFileSync(WASM_PATH);
		const imports = {
			env: {
				abort: (msg: number, file: number, line: number, col: number) => {
					console.error(`WASM Abort: ${line}:${col}`);
				},
				seed: () => Math.random(), // Required by AS Math.random()
			},
		};

		const result = await loader.instantiate(wasmBuffer, imports);
		this.wasmModule = result.exports;
		this.instance = result.instance;
		// Get the exported memory
		this.memory = this.wasmModule.memory as WebAssembly.Memory;

		this.wasmModule.init_memory();

		// Apply Config if provided
		if (config) {
			console.log("Applying WASM Config:", config);
			this.wasmModule.set_config(config.dim, config.m, config.ef);
		}

		// Initialize the index with capacity
		this.wasmModule.init_index(10000);

		console.log(
			"WASM Initialized. Memory size:",
			this.memory.buffer.byteLength,
		);
	}

	// Helper to write a float array to WASM memory
	// Returns the offset
	writeVector(vector: Float32Array): number {
		// In a real app, we would have a dedicated input buffer area in WASM
		// For now, we use a simple alloc from the module if exposed, or just
		// poke into a known free area.
		// A better way: Expose an 'alloc_input(size)' function in WASM.

		// Let's assume we added 'alloc' to exports in index.ts (we need to do that)
		const ptr = this.wasmModule.alloc(vector.length * 4);

		// Create a view on the memory
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
		// Now returns count of results found
		const count = this.wasmModule.search(ptr, k);

		if (count === 0) return [];

		const resultsPtr = this.wasmModule.get_results_ptr();
		const results = [];

		// Read from memory: [id(i32), dist(f32)]
		// We use DataView for safe endianness access, or typed array
		const view = new DataView(this.memory.buffer, resultsPtr, count * 8);

		for (let i = 0; i < count; i++) {
			const id = view.getInt32(i * 8, true); // true = little endian (WASM standard)
			const dist = view.getFloat32(i * 8 + 4, true);
			results.push({ id, dist });
		}

		return results;
	}

	async save(filepath: string) {
		// 1. 获取内存使用
		const usedBytes = this.wasmModule.get_memory_usage();

		// 2. 获取全局状态
		const statePtr = this.wasmModule.get_state_ptr();
		const stateView = new Uint8Array(this.memory.buffer, statePtr, 28);

		// 3. 创建整体 buffer: [state(28B) | memory]
		const memoryView = new Uint8Array(this.memory.buffer, 0, usedBytes);
		const combined = new Uint8Array(stateView.length + memoryView.length);
		combined.set(stateView, 0);
		combined.set(memoryView, stateView.length);

		await fs.promises.writeFile(filepath, combined);
		console.log(
			`Saved ${combined.byteLength} bytes (state + memory) to ${filepath}`,
		);
	}

	async load(filepath: string) {
		if (!fs.existsSync(filepath)) {
			console.log("No save file found.");
			return;
		}

		const buffer = await fs.promises.readFile(filepath);
		console.log(`Loading ${buffer.byteLength} bytes from ${filepath}...`);

		// 28字节是 state
		const stateSize = 28;
		const memorySize = buffer.byteLength - stateSize;

		// 1. 确保内存足够
		const currentSize = this.memory.buffer.byteLength;
		if (currentSize < memorySize) {
			const pagesNeeded = Math.ceil((memorySize - currentSize) / 65536);
			this.memory.grow(pagesNeeded);
		}

		// 2. 写入 WASM 内存
		const wasmView = new Uint8Array(this.memory.buffer);
		wasmView.set(buffer.subarray(stateSize), 0);

		// 3. 恢复状态
		const statePtr = this.wasmModule.alloc(stateSize);
		const stateBytes = buffer.subarray(0, stateSize);
		wasmView.set(stateBytes, statePtr);
		this.wasmModule.set_state_ptr(statePtr);

		// 4. 恢复 allocator
		this.wasmModule.set_memory_usage(memorySize);

		console.log("Loaded state + memory into WASM.");
	}
}
