// src/core/wasm-bridge.ts
import fs from "fs";
import loader from "@assemblyscript/loader";
import path from "path";

const WASM_PATH = path.join(__dirname, "../../build/release.wasm");

export class WasmBridge {
	private wasmModule: any;
	private memory!: WebAssembly.Memory;
	private instance!: WebAssembly.Instance;

	private currentConfig: {
		dim: number;
		m: number;
		ef: number;
		efSearch?: number;
	} | null = null;

	constructor() {}

	async init(config?: {
		dim: number;
		m: number;
		ef: number;
		efSearch?: number;
	}) {
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
			this.currentConfig = config;
			this.wasmModule.set_config(config.dim, config.m, config.ef);
			if (
				typeof this.wasmModule.set_search_config === "function" &&
				config.efSearch
			) {
				this.wasmModule.set_search_config(config.efSearch);
			}
		}

		this.wasmModule.init_index(10000);
		console.log(
			"WASM Initialized. Memory size:",
			this.memory.buffer.byteLength,
		);
	}

	writeVectorI8(vector: Int8Array): number {
		const ptr = this.wasmModule.alloc(vector.length);
		const view = new Int8Array(this.memory.buffer, ptr, vector.length);
		view.set(vector);
		return ptr;
	}

	/**
	 * ✅ 兼容：旧 wasm 可能还没导出 has_node（比如你忘了重新 build wasm）
	 * - 若存在 has_node：精确判断
	 * - 若不存在：保守返回 false（会走 insert，至少不再报错）
	 */
	hasNode(id: number): boolean {
		if (typeof this.wasmModule.has_node === "function") {
			return !!this.wasmModule.has_node(id);
		}
		return false;
	}

	insert(id: number, vectorI8: Int8Array) {
		const ptr = this.writeVectorI8(vectorI8);
		this.wasmModule.insert(id, ptr);
	}

	updateVector(id: number, vectorI8: Int8Array) {
		const ptr = this.writeVectorI8(vectorI8);
		this.wasmModule.update_vector(id, ptr);
	}

	search(vectorI8: Int8Array, k: number) {
		const ptr = this.writeVectorI8(vectorI8);
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

	async save(filePath: string) {
		if (typeof this.wasmModule.get_index_dump_size !== "function") {
			throw new Error("WASM export get_index_dump_size() not found.");
		}

		const dumpSize: number = this.wasmModule.get_index_dump_size();
		const ptr: number = this.wasmModule.alloc(dumpSize);
		const used: number = this.wasmModule.save_index(ptr);

		const view = new Uint8Array(this.memory.buffer, ptr, used);
		await fs.promises.writeFile(filePath, view);

		console.log(`Index saved: ${used} bytes`);
	}

	async load(filePath: string) {
		if (!fs.existsSync(filePath)) return;

		const buf = await fs.promises.readFile(filePath);

		if (typeof this.wasmModule.reset_memory === "function") {
			this.wasmModule.reset_memory();
		} else {
			this.wasmModule.init_memory();
		}

		if (this.currentConfig) {
			this.wasmModule.set_config(
				this.currentConfig.dim,
				this.currentConfig.m,
				this.currentConfig.ef,
			);
			if (
				typeof this.wasmModule.set_search_config === "function" &&
				this.currentConfig.efSearch
			) {
				this.wasmModule.set_search_config(this.currentConfig.efSearch);
			}
		}

		const ptr = this.wasmModule.alloc(buf.length);
		new Uint8Array(this.memory.buffer, ptr, buf.length).set(buf);
		this.wasmModule.load_index(ptr, buf.length);

		console.log(`Index loaded: ${buf.length} bytes`);
	}
}
