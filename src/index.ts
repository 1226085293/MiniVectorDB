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

	// ✅ new: results cap for wasm output buffer
	resultsCap?: number;
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
			await fd.read(buf, 0, bytes, pos);

			for (let j = 0; j < count; j++) {
				const id = start + j;
				const off = j * bpv;
				if (off + bpv > buf.length) continue;

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
		if (this.isReady) return;

		const dim = this.dim();
		const m = this.config.m || Number(process.env.HNSW_M ?? 0) || 16;
		const ef =
			this.config.ef_construction || Number(process.env.HNSW_EF ?? 0) || 100;
		const efSearch =
			this.config.ef_search || Number(process.env.HNSW_EF_SEARCH ?? 0) || 50;

		const capacity =
			this.config.capacity ||
			Number(process.env.HNSW_CAPACITY ?? 0) ||
			1_200_000;

		const seedFromEnv = process.env.HNSW_SEED
			? Number(process.env.HNSW_SEED) >>> 0
			: undefined;
		const seed =
			this.config.seed != null ? this.config.seed >>> 0 : seedFromEnv;

		const resultsCap =
			(this.config.resultsCap ?? Number(process.env.HNSW_RESULTS_CAP ?? 0)) ||
			1000;

		await this.wasm.init({ dim, m, ef, efSearch, capacity, seed, resultsCap });
		await this.meta.ready();
		await this.ensureVectorStoreReady();

		this.isReady = true;
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

	async insert(item: InsertItem): Promise<void> {
		await this.insertMany([item]);
	}

	/**
	 * ✅ Bulk insert optimized + safe internal_id allocation
	 */
	async insertMany(items: InsertItem[]): Promise<void> {
		if (!this.isReady) await this.init();
		if (items.length === 0) return;

		const expectedDim = this.dim();

		const externalIds = items.map((it) => it.id);
		const existingMap = this.meta.getMany(externalIds);

		// count new items first to allocate consecutive internal ids safely
		let newItemsCount = 0;
		for (const it of items) if (!existingMap.get(it.id)) newItemsCount++;

		const newStartId =
			newItemsCount > 0 ? this.meta.allocInternalIds(newItemsCount) : 0;
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

		this.meta.beginBulk();
		try {
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

			// 3) single meta write
			this.meta.addMany(metaEntries, existingMap);

			// 4) write all NEW ids as contiguous runs
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

			// 5) updates: write individually
			for (let i = 0; i < items.length; i++) {
				if (!isNew[i]) {
					await this.writeF32Vector(internalIds[i], f32s[i]);
				}
			}

			// 6) wasm graph ops (async + locked inside bridge)
			for (let i = 0; i < items.length; i++) {
				const internalId = internalIds[i];
				const q8 = q8s[i];

				if (await this.wasm.hasNode(internalId)) {
					await this.wasm.updateVector(internalId, q8);
				} else {
					await this.wasm.insert(internalId, q8);
				}
			}
		} finally {
			await this.meta.endBulk();
		}
	}

	async search(
		query: number[] | Float32Array | string | Buffer | Uint8Array,
		k: number = 10,
		filter?: any,
	): Promise<SearchResult[]> {
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
		const annK = Math.max(k * mult, k);

		const raw = await this.wasm.search(qI8, annK);
		if (raw.length === 0) return [];

		const candidates: number[] = [];
		for (const r of raw) {
			const item = this.meta.getByInternalId(r.id);
			if (!item) continue;

			if (filter) {
				let match = true;
				for (const key in filter) {
					if (item.metadata[key] !== filter[key]) {
						match = false;
						break;
					}
				}
				if (!match) continue;
			}

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
	}

	/**
	 * ✅ Expose ANN-stage raw results (WASM HNSW on Int8 L2^2) for benchmarking.
	 */
	async searchAnnI8(
		query: number[] | Float32Array | string | Buffer | Uint8Array,
		annK: number,
	): Promise<{ internalId: number; dist: number }[]> {
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
		const raw = await this.wasm.search(qI8, annK);

		return raw.map((r) => ({ internalId: r.id, dist: r.dist }));
	}

	async save(filepath: string) {
		await this.wasm.save(filepath);
		if (this.vecFd) await this.vecFd.sync();
	}

	async load(filepath: string) {
		if (!this.isReady) await this.init();
		await this.wasm.load(filepath);
		await this.ensureVectorStoreReady();

		if (this.config.preloadVectors && fs.existsSync(this.vecPath)) {
			this.vecCache = await fs.promises.readFile(this.vecPath);
		}
	}

	getStats() {
		return {
			items: this.meta.items.count(),
			vectorStore: this.vecPath,
			capacity: this.config.capacity,
			metaPath: this.meta.getPath(),
			preloadVectors: !!this.config.preloadVectors,
		};
	}

	async close() {
		await this.meta.close();
		if (this.vecFd) {
			await this.vecFd.close();
			this.vecFd = null;
		}
		this.vecCache = null;
	}
}
