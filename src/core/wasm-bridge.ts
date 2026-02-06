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

	private scratchI8Ptr: number = 0;
	private scratchI8Cap: number = 0;

	// ✅ reusable dump buffer to avoid alloc-leak on repeated load()
	private scratchDumpPtr: number = 0;
	private scratchDumpCap: number = 0;

	private lock: Promise<void> = Promise.resolve();

	private wasmMaxEf: number = 0;

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
				abort: (_msg: number, _file: number, line: number, col: number) => {
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

			// read wasm max ef early
			if (typeof this.wasmModule.get_max_ef === "function") {
				this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
			} else {
				this.wasmMaxEf = 0;
			}

			if (config) {
				this.currentConfig = config;
				this.wasmModule.set_config(config.dim, config.m, config.ef);

				if (
					typeof this.wasmModule.set_search_config === "function" &&
					config.efSearch
				) {
					this.wasmModule.set_search_config(config.efSearch);
				}

				if (typeof this.wasmModule.set_results_cap === "function") {
					const cap = this.clampResultsCap(config.resultsCap ?? 1000);
					this.wasmModule.set_results_cap(cap);
				}
			}

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

			if (config?.dim) {
				this.ensureScratchI8(config.dim);
			}

			// dump buffer reset (will be allocated on demand)
			this.scratchDumpPtr = 0;
			this.scratchDumpCap = 0;

			console.log(
				"WASM Initialized. Memory size:",
				this.memory.buffer.byteLength,
			);
		});
	}

	private clampResultsCap(cap: number): number {
		if (!cap || cap <= 0) return 1;
		if (this.wasmMaxEf > 0) return Math.min(cap | 0, this.wasmMaxEf);
		return cap | 0;
	}

	async getMaxEf(): Promise<number> {
		return this.withLock(() => {
			if (this.wasmMaxEf > 0) return this.wasmMaxEf;
			if (typeof this.wasmModule?.get_max_ef === "function") {
				this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
				return this.wasmMaxEf;
			}
			return 0;
		});
	}

	private ensureScratchI8(nBytes: number) {
		if (nBytes <= 0) throw new Error(`Invalid scratch size: ${nBytes}`);
		if (this.scratchI8Ptr !== 0 && this.scratchI8Cap >= nBytes) return;

		const ptr = this.wasmModule.alloc(nBytes);
		this.scratchI8Ptr = ptr;
		this.scratchI8Cap = nBytes;
	}

	private ensureScratchDump(nBytes: number) {
		if (nBytes <= 0) throw new Error(`Invalid dump scratch size: ${nBytes}`);
		if (this.scratchDumpPtr !== 0 && this.scratchDumpCap >= nBytes) return;

		const ptr = this.wasmModule.alloc(nBytes);
		this.scratchDumpPtr = ptr;
		this.scratchDumpCap = nBytes;
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

	private writeDumpReusable(buf: Buffer): number {
		this.ensureScratchDump(buf.length);
		new Uint8Array(this.memory.buffer, this.scratchDumpPtr, buf.length).set(
			buf,
		);
		return this.scratchDumpPtr;
	}

	async getResultsCap(): Promise<number> {
		return this.withLock(() => {
			if (typeof this.wasmModule.get_results_cap === "function") {
				return Number(this.wasmModule.get_results_cap());
			}
			return 0;
		});
	}

	async setResultsCap(cap: number): Promise<void> {
		await this.withLock(() => {
			if (typeof this.wasmModule.set_results_cap === "function") {
				this.wasmModule.set_results_cap(this.clampResultsCap(cap));
			}
		});
	}

	async getMaxElements(): Promise<number> {
		return this.withLock(() => {
			if (typeof this.wasmModule.get_max_elements === "function") {
				return Number(this.wasmModule.get_max_elements());
			}
			return Number(this.currentConfig?.capacity ?? 0);
		});
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

			if (typeof this.wasmModule.update_and_reconnect === "function") {
				this.wasmModule.update_and_reconnect(id, ptr);
				return;
			}

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

			// ✅ avoid bump-allocator leak on repeated save()
			const mark =
				typeof this.wasmModule.get_memory_usage === "function"
					? Number(this.wasmModule.get_memory_usage()) >>> 0
					: 0;

			const dumpSize: number = this.wasmModule.get_index_dump_size();
			const ptr: number = this.wasmModule.alloc(dumpSize);
			const used: number = this.wasmModule.save_index(ptr);

			const view = new Uint8Array(this.memory.buffer, ptr, used);
			await fs.promises.writeFile(filePath, view);

			if (mark && typeof this.wasmModule.set_memory_usage === "function") {
				this.wasmModule.set_memory_usage(mark);
			}

			console.log(`Index saved: ${used} bytes`);
		});
	}

	async load(filePath: string) {
		await this.withLock(async () => {
			if (!fs.existsSync(filePath)) return;

			const buf = await fs.promises.readFile(filePath);

			const reinitFresh = () => {
				if (typeof this.wasmModule.reset_memory === "function") {
					this.wasmModule.reset_memory();
				} else {
					this.wasmModule.init_memory();
				}

				this.scratchI8Ptr = 0;
				this.scratchI8Cap = 0;
				this.scratchDumpPtr = 0;
				this.scratchDumpCap = 0;

				if (typeof this.wasmModule.get_max_ef === "function") {
					this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
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

					if (typeof this.wasmModule.set_results_cap === "function") {
						this.wasmModule.set_results_cap(
							this.clampResultsCap(this.currentConfig.resultsCap ?? 1000),
						);
					}

					if (typeof this.wasmModule.seed_rng === "function") {
						const seedFromEnv = process.env.HNSW_SEED
							? Number(process.env.HNSW_SEED) >>> 0
							: 0;

						const seed =
							this.currentConfig.seed != null
								? this.currentConfig.seed >>> 0
								: seedFromEnv || (Date.now() ^ (process.pid << 16)) >>> 0;

						this.wasmModule.seed_rng(seed);
					}

					const cap = this.currentConfig.capacity ?? 0;
					this.wasmModule.init_index(cap);

					this.ensureScratchI8(this.currentConfig.dim);
				}
			};

			try {
				if (typeof this.wasmModule.reset_memory === "function") {
					this.wasmModule.reset_memory();
				} else {
					this.wasmModule.init_memory();
				}

				this.scratchI8Ptr = 0;
				this.scratchI8Cap = 0;
				this.scratchDumpPtr = 0;
				this.scratchDumpCap = 0;

				if (typeof this.wasmModule.get_max_ef === "function") {
					this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
				}

				if (this.currentConfig) {
					// set_config is allowed to be idempotent after init (same values)
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
							this.clampResultsCap(this.currentConfig.resultsCap ?? 1000),
						);
					}

					if (typeof this.wasmModule.seed_rng === "function") {
						const seedFromEnv = process.env.HNSW_SEED
							? Number(process.env.HNSW_SEED) >>> 0
							: 0;

						const seed =
							this.currentConfig.seed != null
								? this.currentConfig.seed >>> 0
								: seedFromEnv || (Date.now() ^ (process.pid << 16)) >>> 0;

						this.wasmModule.seed_rng(seed);
					}
				}

				// ✅ reusable dump buffer (no cumulative alloc growth)
				const ptr = this.writeDumpReusable(buf);

				const ok: number = this.wasmModule.load_index(ptr, buf.length);
				if (!ok) {
					throw new Error(
						"WASM load_index failed (format/config mismatch or corrupt dump).",
					);
				}

				if (this.currentConfig?.dim) {
					this.ensureScratchI8(this.currentConfig.dim);
				}

				console.log(`Index loaded: ${buf.length} bytes`);
			} catch (e) {
				// ✅ keep module usable even if load fails
				reinitFresh();
				throw e;
			}
		});
	}
}
