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

	// ✅ 新增：WASM HNSW capacity（必须 >= 预计最大条目数）
	capacity?: number;
}

export class MiniVectorDB {
	config: DBConfig;
	private wasm: WasmBridge;
	private meta: MetaDB;
	private embedder: LocalEmbedder;
	private isReady: boolean = false;

	private vecPath: string;
	private vecFd: fs.promises.FileHandle | null = null;

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

	private async ensureVectorStoreReady(): Promise<void> {
		if (this.vecFd) return;

		const dir = path.dirname(this.vecPath);
		if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

		const exists = fs.existsSync(this.vecPath);
		this.vecFd = await fs.promises.open(this.vecPath, exists ? "r+" : "w+");
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

	private async writeF32Vector(
		internalId: number,
		v: Float32Array,
	): Promise<void> {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const dim = this.config.dim || Number(process.env.VECTOR_DIM ?? 0) || 384;
		const bytesPerVec = dim * 4;
		const pos = internalId * bytesPerVec;

		const buf = Buffer.from(v.buffer, v.byteOffset, v.byteLength);
		const { bytesWritten } = await fd.write(buf, 0, bytesPerVec, pos);
		if (bytesWritten !== bytesPerVec) {
			throw new Error(
				`Vector store write failed for id=${internalId}. bytesWritten=${bytesWritten}`,
			);
		}
	}

	private async readF32Vector(internalId: number): Promise<Float32Array> {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const dim = this.config.dim || Number(process.env.VECTOR_DIM ?? 0) || 384;
		const bytesPerVec = dim * 4;
		const pos = internalId * bytesPerVec;

		const buf = Buffer.allocUnsafe(bytesPerVec);
		const { bytesRead } = await fd.read(buf, 0, bytesPerVec, pos);
		if (bytesRead !== bytesPerVec) {
			throw new Error(
				`Vector store read failed for id=${internalId}. bytesRead=${bytesRead}`,
			);
		}

		const out = new Float32Array(dim);
		const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
		for (let i = 0; i < dim; i++) out[i] = dv.getFloat32(i * 4, true);
		return out;
	}

	async init() {
		if (this.isReady) return;

		const dim = this.config.dim || Number(process.env.VECTOR_DIM ?? 0) || 384;
		const m = this.config.m || Number(process.env.HNSW_M ?? 0) || 16;
		const ef =
			this.config.ef_construction || Number(process.env.HNSW_EF ?? 0) || 100;

		const efSearch =
			this.config.ef_search || Number(process.env.HNSW_EF_SEARCH ?? 0) || 50;

		// ✅ capacity：默认给 1_200_000（足够跑 1M + buffer）
		// 你跑 perf_benchmark 时建议显式传入 capacity=N
		const capacity =
			this.config.capacity ||
			Number(process.env.HNSW_CAPACITY ?? 0) ||
			1_200_000;

		await this.wasm.init({ dim, m, ef, efSearch, capacity });
		await this.meta.ready();
		await this.ensureVectorStoreReady();

		this.isReady = true;
	}

	async insert(item: InsertItem): Promise<void> {
		if (!this.isReady) await this.init();
		const { id, vector, metadata } = item;

		let f32Vec: Float32Array;
		if (
			typeof vector === "string" ||
			vector instanceof Buffer ||
			vector instanceof Uint8Array
		) {
			f32Vec = await this.embedder.embed(vector);
		} else if (vector instanceof Float32Array) {
			f32Vec = vector;
		} else {
			f32Vec = new Float32Array(vector);
		}

		const expectedDim = this.config.dim || 384;
		if (f32Vec.length !== expectedDim) {
			throw new Error(
				`Vector dimension mismatch. Expected ${expectedDim}, got ${f32Vec.length}`,
			);
		}

		const q8 = this.quantizeToI8(f32Vec);

		const existing = this.meta.get(id);
		if (existing) {
			const internalId = existing.internal_id;

			await this.writeF32Vector(internalId, f32Vec);

			if (!this.wasm.hasNode(internalId)) {
				this.wasm.insert(internalId, q8);
			} else {
				this.wasm.updateVector(internalId, q8);
			}

			this.meta.add(id, internalId, metadata || existing.metadata);
			return;
		}

		const internalId = this.meta.items.count();
		this.meta.add(id, internalId, metadata || {});
		await this.writeF32Vector(internalId, f32Vec);
		this.wasm.insert(internalId, q8);
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

		const expectedDim = this.config.dim || 384;
		if (qF32.length !== expectedDim) {
			throw new Error(
				`Query vector dimension mismatch. Expected ${expectedDim}, got ${qF32.length}`,
			);
		}

		const qI8 = this.quantizeToI8(qF32);

		const mult = this.config.rerankMultiplier ?? 30;
		const annK = Math.max(k * mult, k);

		const raw = this.wasm.search(qI8, annK);
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

		const vecs = await Promise.all(
			candidates.map((id) => this.readF32Vector(id)),
		);

		const reranked: { internalId: number; score: number }[] = [];
		for (let i = 0; i < candidates.length; i++) {
			const internalId = candidates[i];
			const v = vecs[i];
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

	async save(filepath: string) {
		await this.wasm.save(filepath);
		if (this.vecFd) await this.vecFd.sync();
	}

	async load(filepath: string) {
		if (!this.isReady) await this.init();
		await this.wasm.load(filepath);
		await this.ensureVectorStoreReady();
	}

	getStats() {
		return {
			memory: (this.wasm as any)["memory"]?.buffer?.byteLength,
			items: this.meta.items.count(),
			vectorStore: this.vecPath,
			capacity: this.config.capacity,
		};
	}

	async close() {
		await this.meta.close();
		if (this.vecFd) {
			await this.vecFd.close();
			this.vecFd = null;
		}
	}
}
