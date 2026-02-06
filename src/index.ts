// src/index.ts
import { WasmBridge } from "./core/wasm-bridge";
import { MetaDB } from "./storage/meta-db";
import { LocalEmbedder } from "./embedder";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";

dotenv.config({ path: path.join(__dirname, "../.env") });

export interface SearchResult {
	id: string;
	score: number;
	metadata: any;
}

export interface InsertItem {
	id: string;
	vector: number[] | Float32Array | string | Buffer | Uint8Array;
	metadata?: any;
}

export interface DBConfig {
	dim?: number;
	modelName?: string;
	modelArchitecture?: "text" | "clip";
	m?: number;
	ef_construction?: number;
	ef_search?: number;

	metaDbPath?: string;
	vectorStorePath?: string;

	rerankMultiplier?: number;

	capacity?: number;
	preloadVectors?: boolean;

	seed?: number;

	resultsCap?: number;

	// Optional cap for ANN candidates to avoid runaway memory/time
	maxAnnK?: number;
}

export class MiniVectorDB {
	config: DBConfig;
	private wasm: WasmBridge;
	private meta: MetaDB;
	private embedder: LocalEmbedder;
	private isReady: boolean = false;

	private vecPath: string;
	private vecFd: fs.promises.FileHandle | null = null;
	private vecCache: Buffer | null = null;

	// ✅ align host caps with wasm MAX_EF
	private wasmMaxEf: number = 0;

	// ✅ Global mutex to avoid meta/file/wasm divergence under concurrency
	private opLock: Promise<void> = Promise.resolve();
	private async withLock<T>(fn: () => Promise<T> | T): Promise<T> {
		let release!: () => void;
		const next = new Promise<void>((r) => (release = r));
		const prev = this.opLock;
		this.opLock = prev.then(() => next);
		await prev;
		try {
			return await fn();
		} finally {
			release();
		}
	}

	constructor(config: DBConfig = {}) {
		this.config = config;
		this.wasm = new WasmBridge();
		this.meta = new MetaDB(config.metaDbPath);
		this.embedder = new LocalEmbedder(
			config.modelName || process.env.MODEL_NAME || "Xenova/all-MiniLM-L6-v2",
			config.modelArchitecture,
		);

		this.vecPath =
			config.vectorStorePath || path.join(__dirname, "../data/vectors.f32.bin");
	}

	private dim(): number {
		return this.config.dim || Number(process.env.VECTOR_DIM ?? 0) || 384;
	}

	private bytesPerVec(): number {
		return this.dim() * 4;
	}

	private capacity(): number {
		return (
			this.config.capacity ||
			Number(process.env.HNSW_CAPACITY ?? 0) ||
			1_200_000
		);
	}

	private maxAnnK(): number {
		const fromCfg = this.config.maxAnnK;
		const fromEnv = process.env.MAX_ANN_K ? Number(process.env.MAX_ANN_K) : 0;
		let cap = (fromCfg ?? fromEnv) || 10_000;

		// ✅ align with wasm MAX_EF (otherwise wasted + confusing)
		if (this.wasmMaxEf > 0) cap = Math.min(cap, this.wasmMaxEf);

		// also reasonable hard floor
		if (cap <= 0) cap = 1;
		return cap;
	}

	private async ensureVectorStoreReady(): Promise<void> {
		if (this.vecFd) return;

		const dir = path.dirname(this.vecPath);
		if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

		const exists = fs.existsSync(this.vecPath);
		this.vecFd = await fs.promises.open(this.vecPath, exists ? "r+" : "w+");

		if (this.config.preloadVectors && exists) {
			this.vecCache = await fs.promises.readFile(this.vecPath);
		}
	}

	private quantizeToI8(v: Float32Array): Int8Array {
		const out = new Int8Array(v.length);
		for (let i = 0; i < v.length; i++) {
			let x = v[i];
			if (x > 1) x = 1;
			else if (x < -1) x = -1;
			let q = Math.round(x * 127);
			if (q > 127) q = 127;
			else if (q < -127) q = -127;
			out[i] = q;
		}
		return out;
	}

	private l2SqF32(a: Float32Array, b: Float32Array): number {
		let s = 0;
		for (let i = 0; i < a.length; i++) {
			const d = a[i] - b[i];
			s += d * d;
		}
		return s;
	}

	private ensureCacheCapacity(minBytes: number) {
		if (!this.config.preloadVectors) return;
		if (!this.vecCache) {
			this.vecCache = Buffer.allocUnsafe(minBytes);
			this.vecCache.fill(0);
			return;
		}
		if (this.vecCache.length < minBytes) {
			const old = this.vecCache;
			const next = Buffer.allocUnsafe(minBytes);
			old.copy(next, 0, 0, old.length);
			next.fill(0, old.length);
			this.vecCache = next;
		}
	}

	private async writeF32VectorsContiguous(
		startInternalId: number,
		vectors: Float32Array[],
	): Promise<void> {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const bpv = this.bytesPerVec();
		const totalBytes = vectors.length * bpv;
		const pos = startInternalId * bpv;

		const buf = Buffer.allocUnsafe(totalBytes);
		for (let i = 0; i < vectors.length; i++) {
			const v = vectors[i];
			const slice = Buffer.from(v.buffer, v.byteOffset, v.byteLength);
			slice.copy(buf, i * bpv, 0, bpv);
		}

		const { bytesWritten } = await fd.write(buf, 0, totalBytes, pos);
		if (bytesWritten !== totalBytes) {
			throw new Error(
				`Vector store bulk write failed. bytesWritten=${bytesWritten}, total=${totalBytes}`,
			);
		}

		if (this.config.preloadVectors) {
			this.ensureCacheCapacity(pos + totalBytes);
			this.vecCache!.set(buf, pos);
		}
	}

	private async writeF32Vector(
		internalId: number,
		v: Float32Array,
	): Promise<void> {
		await this.writeF32VectorsContiguous(internalId, [v]);
	}

	private async readF32Vectors(
		internalIds: number[],
	): Promise<Map<number, Float32Array>> {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const dim = this.dim();
		const bpv = this.bytesPerVec();

		const out = new Map<number, Float32Array>();
		if (internalIds.length === 0) return out;

		if (this.config.preloadVectors && this.vecCache) {
			for (const id of internalIds) {
				const pos = id * bpv;
				if (pos + bpv > this.vecCache.length) continue;
				const dv = new DataView(
					this.vecCache.buffer,
					this.vecCache.byteOffset + pos,
					bpv,
				);
				const v = new Float32Array(dim);
				for (let i = 0; i < dim; i++) v[i] = dv.getFloat32(i * 4, true);
				out.set(id, v);
			}
			return out;
		}

		const ids = Array.from(new Set(internalIds)).sort((a, b) => a - b);

		let i = 0;
		while (i < ids.length) {
			const start = ids[i];
			let end = start;

			while (i + 1 < ids.length && ids[i + 1] === end + 1) {
				i++;
				end = ids[i];
			}

			const count = end - start + 1;
			const pos = start * bpv;
			const bytes = count * bpv;

			const buf = Buffer.allocUnsafe(bytes);
			const { bytesRead } = await fd.read(buf, 0, bytes, pos);

			// ✅ critical: no silent short-read
			if (bytesRead !== bytes) {
				throw new Error(
					`Vector store short read. pos=${pos} need=${bytes} got=${bytesRead}`,
				);
			}

			for (let j = 0; j < count; j++) {
				const id = start + j;
				const off = j * bpv;

				const dv = new DataView(buf.buffer, buf.byteOffset + off, bpv);
				const v = new Float32Array(dim);
				for (let k = 0; k < dim; k++) v[k] = dv.getFloat32(k * 4, true);
				out.set(id, v);
			}

			i++;
		}

		return out;
	}

	async init() {
		return this.withLock(async () => {
			if (this.isReady) return;

			const dim = this.dim();
			const m = this.config.m || Number(process.env.HNSW_M ?? 0) || 16;
			const ef =
				this.config.ef_construction || Number(process.env.HNSW_EF ?? 0) || 100;
			const efSearch =
				this.config.ef_search || Number(process.env.HNSW_EF_SEARCH ?? 0) || 50;

			const capacity = this.capacity();

			const seedFromEnv = process.env.HNSW_SEED
				? Number(process.env.HNSW_SEED) >>> 0
				: undefined;
			const seed =
				this.config.seed != null ? this.config.seed >>> 0 : seedFromEnv;

			const resultsCap =
				(this.config.resultsCap ?? Number(process.env.HNSW_RESULTS_CAP ?? 0)) ||
				1000;

			await this.wasm.init({
				dim,
				m,
				ef,
				efSearch,
				capacity,
				seed,
				resultsCap,
			});

			// ✅ align host caps with wasm MAX_EF
			this.wasmMaxEf = await this.wasm.getMaxEf();

			// also clamp resultsCap in wasm again (in case env/config > maxEf)
			if (this.wasmMaxEf > 0) {
				const clamp = Math.min(resultsCap, this.wasmMaxEf);
				await this.wasm.setResultsCap(clamp);
			}

			await this.meta.ready();
			await this.ensureVectorStoreReady();

			const wasmCap = await this.wasm.getMaxElements();
			if (wasmCap && wasmCap !== capacity) {
				throw new Error(
					`WASM capacity mismatch. expected=${capacity}, wasm=${wasmCap}`,
				);
			}

			this.isReady = true;
		});
	}

	private async resolveVectorToF32(
		vector: InsertItem["vector"],
	): Promise<Float32Array> {
		if (
			typeof vector === "string" ||
			vector instanceof Buffer ||
			vector instanceof Uint8Array
		) {
			return await this.embedder.embed(vector);
		}
		if (vector instanceof Float32Array) return vector;
		return new Float32Array(vector);
	}

	private ensureWithinCapacity(internalId: number) {
		const cap = this.capacity();
		if (internalId < 0 || internalId >= cap) {
			throw new Error(
				`Internal ID out of capacity. id=${internalId}, capacity=${cap}. ` +
					`Increase HNSW_CAPACITY / config.capacity.`,
			);
		}
	}

	private ensureAllocWithinCapacity(start: number, n: number) {
		const cap = this.capacity();
		const endExclusive = start + n;
		if (start < 0 || n <= 0 || endExclusive > cap) {
			throw new Error(
				`Capacity overflow. need [${start}, ${endExclusive}) but capacity=${cap}. ` +
					`Increase HNSW_CAPACITY / config.capacity.`,
			);
		}
	}

	async insert(item: InsertItem): Promise<void> {
		await this.insertMany([item]);
	}

	async insertMany(items: InsertItem[]): Promise<void> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			if (items.length === 0) return;

			const expectedDim = this.dim();

			const externalIds = items.map((it) => it.id);
			const existingMap = this.meta.getMany(externalIds);

			let newItemsCount = 0;
			for (const it of items) if (!existingMap.get(it.id)) newItemsCount++;

			// ✅ BEGIN BULK (we'll COMMIT only after vectors+wasm succeed)
			this.meta.beginBulk();
			let commit = false;

			try {
				let newStartId = 0;
				if (newItemsCount > 0) {
					newStartId = this.meta.allocInternalIds(newItemsCount);
					this.ensureAllocWithinCapacity(newStartId, newItemsCount);
				}

				let newCursor = 0;

				const f32s: Float32Array[] = new Array(items.length);
				const q8s: Int8Array[] = new Array(items.length);
				const internalIds: number[] = new Array(items.length);
				const isNew: boolean[] = new Array(items.length);

				const metaEntries: {
					external_id: string;
					internal_id: number;
					metadata: any;
				}[] = new Array(items.length);

				// 1) resolve vectors + assign internal ids (NO meta write yet)
				for (let i = 0; i < items.length; i++) {
					const { id, vector, metadata } = items[i];
					const f32 = await this.resolveVectorToF32(vector);

					if (f32.length !== expectedDim) {
						throw new Error(
							`Vector dimension mismatch. Expected ${expectedDim}, got ${f32.length}`,
						);
					}

					const q8 = this.quantizeToI8(f32);

					const existing = existingMap.get(id);
					if (existing) {
						this.ensureWithinCapacity(existing.internal_id);

						internalIds[i] = existing.internal_id;
						isNew[i] = false;

						metaEntries[i] = {
							external_id: id,
							internal_id: existing.internal_id,
							metadata: metadata ?? existing.metadata,
						};
					} else {
						const newInternalId = newStartId + newCursor;
						newCursor++;

						this.ensureWithinCapacity(newInternalId);

						internalIds[i] = newInternalId;
						isNew[i] = true;

						metaEntries[i] = {
							external_id: id,
							internal_id: newInternalId,
							metadata: metadata || {},
						};
					}

					f32s[i] = f32;
					q8s[i] = q8;
				}

				// 2) write vectors (new contiguous runs first)
				{
					const pairs: { id: number; vec: Float32Array }[] = [];
					for (let i = 0; i < items.length; i++)
						if (isNew[i]) pairs.push({ id: internalIds[i], vec: f32s[i] });
					pairs.sort((a, b) => a.id - b.id);

					let p = 0;
					while (p < pairs.length) {
						const start = pairs[p].id;
						const vecs: Float32Array[] = [pairs[p].vec];
						let end = start;

						while (p + 1 < pairs.length && pairs[p + 1].id === end + 1) {
							p++;
							end = pairs[p].id;
							vecs.push(pairs[p].vec);
						}

						await this.writeF32VectorsContiguous(start, vecs);
						p++;
					}
				}

				// existing: write individually
				for (let i = 0; i < items.length; i++) {
					if (!isNew[i]) {
						await this.writeF32Vector(internalIds[i], f32s[i]);
					}
				}

				// ✅ fsync vectors BEFORE committing meta (crash-consistency)
				if (this.vecFd) {
					await this.vecFd.sync();
				}

				// 3) wasm graph ops
				for (let i = 0; i < items.length; i++) {
					const internalId = internalIds[i];
					const q8 = q8s[i];

					if (isNew[i]) {
						await this.wasm.insert(internalId, q8);
					} else {
						if (await this.wasm.hasNode(internalId)) {
							await this.wasm.updateVector(internalId, q8);
						} else {
							await this.wasm.insert(internalId, q8);
						}
					}
				}

				// 4) commit meta LAST
				this.meta.addMany(metaEntries, existingMap);

				commit = true;
			} finally {
				// ✅ commit=true only if fully succeeded; otherwise rollback bulk changes
				await this.meta.endBulk(commit);
			}
		});
	}

	private async ensureResultsCapAtLeast(n: number) {
		if (n <= 0) return;

		// ✅ clamp to wasm MAX_EF
		if (this.wasmMaxEf > 0) {
			n = Math.min(n, this.wasmMaxEf);
		}

		const cur = await this.wasm.getResultsCap();
		if (cur >= n) return;

		// grow (still clamped inside wasm bridge)
		let next = cur > 0 ? cur : 1000;
		while (next < n) next = next * 2;

		await this.wasm.setResultsCap(next);
	}

	async search(
		query: number[] | Float32Array | string | Buffer | Uint8Array,
		k: number = 10,
		filter?: any,
	): Promise<SearchResult[]> {
		// ✅ serialize with writes to avoid reading partially-written vector/meta state
		return this.withLock(async () => {
			if (!this.isReady) await this.init();

			let qF32: Float32Array;
			if (
				typeof query === "string" ||
				query instanceof Buffer ||
				query instanceof Uint8Array
			) {
				qF32 = await this.embedder.embed(query);
			} else if (query instanceof Float32Array) {
				qF32 = query;
			} else {
				qF32 = new Float32Array(query);
			}

			const expectedDim = this.dim();
			if (qF32.length !== expectedDim) {
				throw new Error(
					`Query vector dimension mismatch. Expected ${expectedDim}, got ${qF32.length}`,
				);
			}

			const qI8 = this.quantizeToI8(qF32);

			const mult = this.config.rerankMultiplier ?? 30;
			let annK = Math.max(k * mult, k);

			// ✅ cap ANN candidates to avoid runaway (and align with wasm MAX_EF)
			const cap = this.maxAnnK();
			if (annK > cap) annK = cap;

			await this.ensureResultsCapAtLeast(annK);

			const allowedSet = filter ? this.meta.filterInternalIdSet(filter) : null;

			const raw = await this.wasm.search(qI8, annK);
			if (raw.length === 0) return [];

			const candidates: number[] = [];
			for (const r of raw) {
				if (allowedSet && !allowedSet.has(r.id)) continue;

				const item = this.meta.getByInternalId(r.id);
				if (!item) continue;

				candidates.push(r.id);
				if (candidates.length >= annK) break;
			}
			if (candidates.length === 0) return [];

			const vecMap = await this.readF32Vectors(candidates);

			const reranked: { internalId: number; score: number }[] = [];
			for (const internalId of candidates) {
				const v = vecMap.get(internalId);
				if (!v) continue;
				reranked.push({ internalId, score: this.l2SqF32(qF32, v) });
			}

			reranked.sort((a, b) => a.score - b.score);

			const results: SearchResult[] = [];
			for (const r of reranked) {
				const item = this.meta.getByInternalId(r.internalId);
				if (!item) continue;

				results.push({
					id: item.external_id,
					score: r.score,
					metadata: item.metadata,
				});
				if (results.length >= k) break;
			}

			return results;
		});
	}

	async save(filepath: string) {
		return this.withLock(async () => {
			await this.wasm.save(filepath);
			if (this.vecFd) await this.vecFd.sync();
		});
	}

	async load(filepath: string) {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			await this.wasm.load(filepath);
			await this.ensureVectorStoreReady();

			if (this.config.preloadVectors && fs.existsSync(this.vecPath)) {
				this.vecCache = await fs.promises.readFile(this.vecPath);
			}
		});
	}

	getStats() {
		return {
			items: this.meta.items.count(),
			vectorStore: this.vecPath,
			capacity: this.config.capacity,
			metaPath: this.meta.getPath(),
			preloadVectors: !!this.config.preloadVectors,
			wasmMaxEf: this.wasmMaxEf || undefined,
		};
	}

	async close() {
		return this.withLock(async () => {
			await this.meta.close();
			if (this.vecFd) {
				await this.vecFd.close();
				this.vecFd = null;
			}
			this.vecCache = null;
		});
	}
}
