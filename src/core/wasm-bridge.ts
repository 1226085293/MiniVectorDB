// src/core/wasm-bridge.ts
import fs from "fs";
import loader from "@assemblyscript/loader";
import path from "path";

const WASM_PATH = path.join(__dirname, "../../build/release.wasm");

export class WasmBridge {
	private wasmModule: any;
	private memory!: WebAssembly.Memory;

	private currentConfig: {
		dim: number;
		m: number;
		ef: number;
		efSearch?: number;
		capacity?: number;
		seed?: number;
		resultsCap?: number;
	} | null = null;

	// ✅ Reusable scratch buffer inside WASM linear memory (Int8)
	private scratchI8Ptr: number = 0;
	private scratchI8Cap: number = 0;

	// ✅ serialize all wasm interactions (future-proof)
	private lock: Promise<void> = Promise.resolve();

	private async withLock<T>(fn: () => T | Promise<T>): Promise<T> {
		let release!: () => void;
		const next = new Promise<void>((r) => (release = r));
		const prev = this.lock;
		this.lock = prev.then(() => next);
		await prev;
		try {
			return await fn();
		} finally {
			release();
		}
	}

	async init(config?: {
		dim: number;
		m: number;
		ef: number;
		efSearch?: number;
		capacity?: number;
		seed?: number;
		resultsCap?: number;
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
		this.memory = this.wasmModule.memory as WebAssembly.Memory;

		await this.withLock(async () => {
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

				// ✅ results cap
				if (typeof this.wasmModule.set_results_cap === "function") {
					this.wasmModule.set_results_cap(config.resultsCap ?? 1000);
				}
			}

			// ✅ seed RNG in wasm (deterministic if provided)
			if (typeof this.wasmModule.seed_rng === "function") {
				const seedFromEnv = process.env.HNSW_SEED
					? Number(process.env.HNSW_SEED) >>> 0
					: 0;

				const seed =
					config?.seed != null
						? config.seed >>> 0
						: seedFromEnv || (Date.now() ^ (process.pid << 16)) >>> 0;

				this.wasmModule.seed_rng(seed);
			}

			const cap = config?.capacity ?? 10000;
			this.wasmModule.init_index(cap);

			// ✅ allocate scratch once (size = dim bytes, for Int8 vectors)
			if (config?.dim) {
				this.ensureScratchI8(config.dim);
			}

			console.log(
				"WASM Initialized. Memory size:",
				this.memory.buffer.byteLength,
			);
		});
	}

	private ensureScratchI8(nBytes: number) {
		if (nBytes <= 0) throw new Error(`Invalid scratch size: ${nBytes}`);
		if (this.scratchI8Ptr !== 0 && this.scratchI8Cap >= nBytes) return;

		const ptr = this.wasmModule.alloc(nBytes);
		this.scratchI8Ptr = ptr;
		this.scratchI8Cap = nBytes;
	}

	private writeVectorI8Reusable(vector: Int8Array): number {
		this.ensureScratchI8(vector.length);

		const view = new Int8Array(
			this.memory.buffer,
			this.scratchI8Ptr,
			vector.length,
		);
		view.set(vector);
		return this.scratchI8Ptr;
	}

	async hasNode(id: number): Promise<boolean> {
		return this.withLock(() => {
			if (typeof this.wasmModule.has_node === "function")
				return !!this.wasmModule.has_node(id);
			return false;
		});
	}

	async insert(id: number, vectorI8: Int8Array): Promise<void> {
		await this.withLock(() => {
			const ptr = this.writeVectorI8Reusable(vectorI8);
			this.wasmModule.insert(id, ptr);
		});
	}

	async updateVector(id: number, vectorI8: Int8Array): Promise<void> {
		await this.withLock(() => {
			const ptr = this.writeVectorI8Reusable(vectorI8);
			this.wasmModule.update_vector(id, ptr);
		});
	}

	async search(
		vectorI8: Int8Array,
		k: number,
	): Promise<{ id: number; dist: number }[]> {
		return this.withLock(() => {
			const ptr = this.writeVectorI8Reusable(vectorI8);
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
		});
	}

	async save(filePath: string) {
		await this.withLock(async () => {
			if (typeof this.wasmModule.get_index_dump_size !== "function") {
				throw new Error("WASM export get_index_dump_size() not found.");
			}

			const dumpSize: number = this.wasmModule.get_index_dump_size();
			const ptr: number = this.wasmModule.alloc(dumpSize);
			const used: number = this.wasmModule.save_index(ptr);

			const view = new Uint8Array(this.memory.buffer, ptr, used);
			await fs.promises.writeFile(filePath, view);

			console.log(`Index saved: ${used} bytes`);
		});
	}

	async load(filePath: string) {
		await this.withLock(async () => {
			if (!fs.existsSync(filePath)) return;

			const buf = await fs.promises.readFile(filePath);

			// reset allocator (bump pointer) -> all old pointers invalid
			if (typeof this.wasmModule.reset_memory === "function") {
				this.wasmModule.reset_memory();
			} else {
				this.wasmModule.init_memory();
			}

			// scratch becomes invalid
			this.scratchI8Ptr = 0;
			this.scratchI8Cap = 0;

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

				if (typeof this.wasmModule.set_results_cap === "function") {
					this.wasmModule.set_results_cap(
						this.currentConfig.resultsCap ?? 1000,
					);
				}
			}

			// reseed after reset
			if (typeof this.wasmModule.seed_rng === "function") {
				const seedFromEnv = process.env.HNSW_SEED
					? Number(process.env.HNSW_SEED) >>> 0
					: 0;

				const seed =
					this.currentConfig?.seed != null
						? this.currentConfig.seed >>> 0
						: seedFromEnv || (Date.now() ^ (process.pid << 16)) >>> 0;

				this.wasmModule.seed_rng(seed);
			}

			// load dump
			const ptr = this.wasmModule.alloc(buf.length);
			new Uint8Array(this.memory.buffer, ptr, buf.length).set(buf);

			const ok: number = this.wasmModule.load_index(ptr, buf.length);
			if (!ok) {
				throw new Error(
					"WASM load_index failed (format/config mismatch or corrupt dump).",
				);
			}

			// re-create scratch
			if (this.currentConfig?.dim) {
				this.ensureScratchI8(this.currentConfig.dim);
			}

			console.log(`Index loaded: ${buf.length} bytes`);
		});
	}
}
