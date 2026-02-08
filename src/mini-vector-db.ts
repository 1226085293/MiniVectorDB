// src/mini-vector-db.ts
import fs from "fs";
import path from "path";
import readline from "readline";

import { WasmBridge } from "./core/wasm-bridge";
import { MetaDB } from "./storage/meta-db";
import { LocalEmbedder, LocalEmbedderOptions } from "./embedder";

import type {
	DBOpenOptions,
	EmbedderLike,
	ExportJSONLOptions,
	ImportJSONLOptions,
	InsertItem,
	InsertManyOptions,
	InternalResolvedConfig,
	RebuildOptions,
	ScoreMode,
	SearchOptions,
	SearchResult,
	UpdateMetadataOptions,
} from "./types";

import { resolveOpenConfig } from "./config";
import { atomicReplace, makeUniqueTmpPath } from "./utils/fs-atomic";
import { lockKeyOf } from "./utils/locks";

/**
 * MiniVectorDB
 * 迷你向量数据库
 *
 * EN:
 * A lightweight local vector database: it stores embeddings (vectors) on disk,
 * keeps metadata in JSON, and uses a WASM ANN index (HNSW-like) for fast similarity search.
 *
 * CN:
 * 一个轻量的本地向量数据库：将“嵌入向量（embedding）”存盘、元数据存 JSON，
 * 通过 WASM 近似最近邻（ANN，HNSW 类）索引进行快速相似度检索。
 */
export class MiniVectorDB<TMeta = any> {
	/** EN: Resolved internal configuration. / CN: 解析后的内部配置。 */
	public readonly cfg: InternalResolvedConfig;

	private static globalLocks: Map<string, Promise<void>> = new Map();

	private wasm = new WasmBridge();
	private meta: MetaDB;
	private embedder: EmbedderLike;

	private isReady = false;

	private vecPath: string;
	private dumpPath: string;
	private metaPath: string;
	private statePath: string;
	private oplogPath: string;

	private vecFd: fs.promises.FileHandle | null = null;
	private vecCache: Buffer | null = null;

	private wasmMaxEf = 0;
	private capacity: number;

	private embCache = new Map<string, Float32Array>();
	private embCacheMax = 0;

	private deletedSinceRebuild = 0;

	private oplogFd: fs.promises.FileHandle | null = null;
	private opLock: Promise<void> = Promise.resolve();

	private async withLock<T>(fn: () => Promise<T> | T): Promise<T> {
		// ... (internal)
		let releaseLocal!: () => void;
		const nextLocal = new Promise<void>((r) => (releaseLocal = r));
		const prevLocal = this.opLock;
		this.opLock = prevLocal.then(() => nextLocal);
		await prevLocal;

		const gkey = lockKeyOf(this.cfg);
		const prevGlobal = MiniVectorDB.globalLocks.get(gkey) || Promise.resolve();

		let releaseGlobal!: () => void;
		const nextGlobal = new Promise<void>((r) => (releaseGlobal = r));

		MiniVectorDB.globalLocks.set(
			gkey,
			prevGlobal.then(() => nextGlobal),
		);
		await prevGlobal;

		try {
			return await fn();
		} finally {
			releaseGlobal();
			releaseLocal();

			const cur = MiniVectorDB.globalLocks.get(gkey);
			if (cur === nextGlobal) MiniVectorDB.globalLocks.delete(gkey);
		}
	}

	/**
	 * Open (or create) a database instance.
	 * 打开（或创建）数据库实例。
	 *
	 * EN:
	 * - Initializes storage directory, metadata store and vector store.
	 * - Loads ANN index snapshot if present; otherwise it's OK (you can rebuild).
	 *
	 * CN:
	 * - 初始化存储目录、元数据与向量存储。
	 * - 若存在 ANN 索引快照则尝试加载；没有也不报错（可后续 rebuild）。
	 *
	 * @param opts EN: Open options, including storage settings and optional embedder override.
	 *             CN: 打开配置，可包含存储路径/容量等，以及可选的 embedder 覆盖。
	 * @returns EN: Ready-to-use MiniVectorDB instance.
	 *          CN: 可直接使用的 MiniVectorDB 实例。
	 */
	static async open<TMeta = any>(
		opts: DBOpenOptions = {},
	): Promise<MiniVectorDB<TMeta>> {
		const cfg = resolveOpenConfig(opts);
		const db = new MiniVectorDB<TMeta>(cfg, opts.embedder);
		await db.init();
		try {
			await db.load();
		} catch {
			// ignore
		}
		return db;
	}

	private constructor(
		cfg: InternalResolvedConfig,
		embedderOverride?: EmbedderLike,
	) {
		// ... (internal)
		this.cfg = cfg;
		this.capacity = cfg.capacity;

		if (!fs.existsSync(cfg.storageDir))
			fs.mkdirSync(cfg.storageDir, { recursive: true });

		this.metaPath = path.join(cfg.storageDir, `${cfg.prefix}metadata.json`);
		this.vecPath = path.join(cfg.storageDir, `${cfg.prefix}vectors.f32.bin`);
		this.dumpPath = path.join(cfg.storageDir, `${cfg.prefix}dump.bin`);
		this.statePath = path.join(cfg.storageDir, `${cfg.prefix}state.json`);
		this.oplogPath = path.join(cfg.storageDir, `${cfg.prefix}ann.oplog`);

		this.meta = new MetaDB(this.metaPath);

		if (embedderOverride) {
			this.embedder = embedderOverride;
		} else {
			const leOpts: LocalEmbedderOptions = {
				cacheDir: cfg.modelCacheDir,
				localFilesOnly: cfg.localFilesOnly,
			};
			this.embedder = new LocalEmbedder(
				cfg.modelName,
				cfg.modelArchitecture,
				leOpts,
			);
		}

		this.embCacheMax = Math.max(0, cfg.embeddingCacheSize | 0);
	}

	/**
	 * Initialize runtime resources.
	 * 初始化运行时资源。
	 *
	 * EN:
	 * Must be called before any operation (insert/search/etc).
	 * This sets up the WASM ANN index, embedder (if needed), metadata DB and file handles.
	 *
	 * CN:
	 * 在任何操作（insert/search 等）前必须完成初始化：
	 * 包括 WASM ANN 索引、embedding 模型（如需要）、元数据与文件句柄等。
	 */
	async init(): Promise<void> {
		return this.withLock(async () => {
			if (this.isReady) return;

			await this.wasm.init({
				dim: this.cfg.dim,
				m: this.cfg.m,
				ef: this.cfg.ef_construction,
				efSearch: this.cfg.baseEfSearch,
				capacity: this.capacity,
				seed: this.cfg.seed,
				resultsCap: this.cfg.resultsCap,
			});

			this.wasmMaxEf = await this.wasm.getMaxEf();
			if (this.embedder.init) await this.embedder.init();

			await this.meta.ready();
			await this.ensureVectorStoreReady();
			await this.ensureOplogReady();

			this.isReady = true;
		});
	}

	/**
	 * Insert one item (id + input + optional metadata).
	 * 插入单条数据（id + 输入 + 可选元数据）。
	 *
	 * EN:
	 * - `input` can be text/binary/vector. If it's text/binary, it will be embedded into a vector.
	 * - Vectors are normalized and stored; metadata is stored separately.
	 *
	 * CN:
	 * - `input` 可为文本/二进制/向量；若为文本/二进制，将先生成 embedding 向量。
	 * - 向量会归一化后写入存储；元数据单独保存。
	 */
	async insert(item: InsertItem<TMeta>): Promise<void> {
		return this.insertMany([item]);
	}

	/**
	 * Insert many items in one batch.
	 * 批量插入多条数据。
	 *
	 * EN:
	 * Batch insert is faster than repeated `insert()`:
	 * - Embeds/normalizes/quantizes vectors.
	 * - Writes vectors to disk (bulk for new IDs).
	 * - Updates ANN index in one WASM call.
	 *
	 * CN:
	 * 批量插入比多次 `insert()` 更快：
	 * - 统一生成/归一化/量化向量；
	 * - 对新 ID 使用连续写入；
	 * - 通过一次 WASM 调用更新 ANN 索引。
	 *
	 * @param items EN: Items to insert.
	 *             CN: 待插入的数据列表。
	 * @param opts  EN: Batch options (e.g. progress callback).
	 *             CN: 批量选项（如进度回调）。
	 */
	async insertMany(
		items: InsertItem<TMeta>[],
		opts: InsertManyOptions = {},
	): Promise<void> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			if (!items.length) return;

			const onProgress = opts.onProgress;
			const expectedDim = this.cfg.dim;

			const externalIds = items.map((it) => it.id);
			const existingMap = this.meta.getMany(externalIds);

			this.meta.beginBulk();
			let commit = false;

			try {
				let newItemsCount = 0;
				for (const it of items) if (!existingMap.get(it.id)) newItemsCount++;

				let newStartId = 0;
				if (newItemsCount > 0) {
					newStartId = this.meta.allocInternalIds(newItemsCount);
					this.ensureAllocWithinCapacity(newStartId, newItemsCount);
				}

				let newCursor = 0;

				const f32s: Float32Array[] = new Array(items.length);
				const q8s: Int8Array[] = new Array(items.length);
				const internalIds: number[] = new Array(items.length);
				const metaEntries: {
					external_id: string;
					internal_id: number;
					metadata: any;
					deleted?: boolean;
				}[] = new Array(items.length);

				for (let i = 0; i < items.length; i++) {
					const { id, input, metadata } = items[i];
					const f32 = await this.resolveToF32(input);

					if (f32.length !== expectedDim) {
						throw new Error(
							`Vector dimension mismatch. Expected ${expectedDim}, got ${f32.length}`,
						);
					}

					const existing = existingMap.get(id);
					if (existing) {
						this.ensureWithinCapacity(existing.internal_id);
						internalIds[i] = existing.internal_id;
						metaEntries[i] = {
							external_id: id,
							internal_id: existing.internal_id,
							metadata: metadata ?? existing.metadata,
							deleted: false,
						};
					} else {
						const newInternalId = newStartId + newCursor;
						newCursor++;
						this.ensureWithinCapacity(newInternalId);

						internalIds[i] = newInternalId;
						metaEntries[i] = {
							external_id: id,
							internal_id: newInternalId,
							metadata: metadata ?? {},
							deleted: false,
						};
					}

					f32s[i] = f32;
					q8s[i] = this.quantizeToI8(f32);

					onProgress?.(i + 1, items.length);
				}

				// ... (rest unchanged)
				{
					const pairs: { id: number; vec: Float32Array }[] = [];
					for (let i = 0; i < items.length; i++) {
						const ex = existingMap.get(items[i].id);
						if (!ex) pairs.push({ id: internalIds[i], vec: f32s[i] });
					}
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

				for (let i = 0; i < items.length; i++) {
					const ex = existingMap.get(items[i].id);
					if (ex) await this.writeF32Vector(internalIds[i], f32s[i]);
				}

				if (this.vecFd) await this.vecFd.sync();

				const pairs = internalIds.map((id, i) => ({ id, vectorI8: q8s[i] }));
				await this.wasm.insertMany(pairs);

				const uniq = Array.from(new Set(internalIds));
				let lines = "";
				for (const id of uniq) lines += `U ${id}\n`;
				await this.appendOplogLocked(lines);

				this.meta.addMany(metaEntries, existingMap);
				commit = true;
			} finally {
				await this.meta.endBulk(commit);
			}
		});
	}

	/**
	 * Soft delete items by IDs (metadata-only).
	 * 软删除：按 ID 删除（仅标记，不立即物理清除）。
	 *
	 * EN:
	 * - Deleted items are excluded from search results.
	 * - Data is not physically removed until a compact rebuild.
	 *
	 * CN:
	 * - 删除后检索会自动过滤掉；
	 * - 真正释放空间需通过 compact rebuild（重建压缩）。
	 *
	 * @returns EN: counters (removed/missing/alreadyDeleted)
	 *          CN: 统计信息（删除/不存在/已删除）
	 */
	async removeMany(
		ids: string[],
	): Promise<{ removed: number; missing: number; alreadyDeleted: number }> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			const r = this.meta.markDeletedMany(ids, true);

			if (r.changed > 0) {
				this.deletedSinceRebuild += r.changed;
				let lines = "";
				for (const iid of r.internalIds) lines += `D ${iid}\n`;
				await this.appendOplogLocked(lines);
			}

			await this.meta.saveNow();
			await this.maybeAutoRebuildLocked();

			return {
				removed: r.changed,
				missing: r.missing,
				alreadyDeleted: r.already,
			};
		});
	}

	/**
	 * Soft delete a single ID.
	 * 删除单条（软删除）。
	 *
	 * @returns EN: true if removed now; false if missing/already deleted.
	 *          CN: 本次是否删除成功（不存在/已删除则为 false）。
	 */
	async remove(id: string): Promise<boolean> {
		const r = await this.removeMany([id]);
		return r.removed > 0;
	}

	/**
	 * Update metadata for an existing item.
	 * 更新某条数据的元数据。
	 *
	 * EN:
	 * `merge=true` (default) merges fields; `merge=false` replaces metadata entirely.
	 *
	 * CN:
	 * `merge=true`（默认）为合并更新；`merge=false` 为整体替换。
	 *
	 * @returns EN: true if updated; false if id not found.
	 *          CN: 是否更新成功（找不到 id 返回 false）。
	 */
	async updateMetadata(
		id: string,
		metadata: any,
		opts: UpdateMetadataOptions = {},
	): Promise<boolean> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			const ok = this.meta.updateMetadata(id, metadata, {
				merge: opts.merge !== false,
			});
			if (ok) await this.meta.saveNow();
			return ok;
		});
	}

	/**
	 * Similarity search (single query).
	 * 相似度检索（单条查询）。
	 *
	 * EN:
	 * - `query` can be text/binary/vector; if text/binary, it will be embedded to a vector.
	 * - Returns topK closest items by ANN + optional exact rerank.
	 * - Score meaning depends on `opts.score`:
	 *   - "l2": smaller is closer (distance)
	 *   - "cosine": larger is more similar [-1..1]
	 *   - "sim": normalized similarity [0..1]
	 *
	 * CN:
	 * - `query` 可为文本/二进制/向量；文本/二进制会先转为 embedding。
	 * - 先用 ANN 粗召回，再用 float 向量精排（rerank）。
	 * - 分数含义取决于 `opts.score`：
	 *   - "l2": 越小越相近（距离）
	 *   - "cosine": 越大越相似 [-1..1]
	 *   - "sim": 归一化相似度 [0..1]
	 *
	 * @param opts EN: topK, score mode, and optional metadata filter.
	 *             CN: topK、计分方式、以及可选的元数据过滤条件。
	 */
	async search(
		query: InsertItem["input"],
		opts: SearchOptions<TMeta> = {},
	): Promise<SearchResult<TMeta>[]> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();

			const topK = Math.max(1, (opts.topK ?? 10) | 0);
			const scoreMode: ScoreMode = opts.score ?? "l2";

			const qF32 = await this.resolveToF32(query);
			if (qF32.length !== this.cfg.dim) {
				throw new Error(
					`Query vector dimension mismatch. Expected ${this.cfg.dim}, got ${qF32.length}`,
				);
			}
			const qI8 = this.quantizeToI8(qF32);

			const efSearch = Math.max(this.cfg.baseEfSearch, topK * 2);
			await this.wasm.setEfSearch(efSearch);

			let annK = Math.max(topK, topK * this.cfg.rerankMultiplier);
			annK = Math.min(annK, this.cfg.maxAnnK);
			if (this.wasmMaxEf > 0) annK = Math.min(annK, this.wasmMaxEf);
			await this.ensureResultsCapAtLeast(annK);

			const raw = await this.wasm.search(qI8, annK);
			if (!raw.length) return [];

			const filter = opts.filter;
			const allowedSet =
				filter && typeof filter === "object" && typeof filter !== "function"
					? this.meta.filterInternalIdSet(filter)
					: null;
			const predicate =
				typeof filter === "function" ? (filter as (m: TMeta) => boolean) : null;

			const candidates: number[] = [];
			for (const r of raw) {
				if (allowedSet && !allowedSet.has(r.id)) continue;
				const item = this.meta.getByInternalId(r.id);
				if (!item || item.deleted) continue;
				if (predicate && !predicate(item.metadata)) continue;
				candidates.push(r.id);
				if (candidates.length >= annK) break;
			}
			if (!candidates.length) return [];

			const vecMap = await this.readF32Vectors(candidates);
			const reranked: { internalId: number; l2: number }[] = [];

			for (const internalId of candidates) {
				const v = vecMap.get(internalId);
				if (!v) continue;
				reranked.push({ internalId, l2: this.l2SqF32(qF32, v) });
			}

			reranked.sort((a, b) => a.l2 - b.l2 || a.internalId - b.internalId);

			const results: SearchResult<TMeta>[] = [];
			for (const r of reranked) {
				const item = this.meta.getByInternalId(r.internalId);
				if (!item || item.deleted) continue;
				results.push({
					id: item.external_id,
					score: this.toScore(r.l2, scoreMode),
					metadata: item.metadata as TMeta,
				});
				if (results.length >= topK) break;
			}
			return results;
		});
	}

	/**
	 * Similarity search (batch queries).
	 * 相似度检索（批量查询）。
	 *
	 * EN:
	 * Executes multiple queries under a single DB lock and a single WASM batch call.
	 * This is typically faster than calling `search()` in a loop.
	 *
	 * CN:
	 * 在一次 DB 锁内完成多个查询，并使用 WASM 批量检索接口；
	 * 通常比循环调用 `search()` 更快。
	 */
	async searchMany(
		queries: InsertItem["input"][],
		opts: SearchOptions<TMeta> = {},
	): Promise<SearchResult<TMeta>[][]> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			if (!queries.length) return [];

			const topK = Math.max(1, (opts.topK ?? 10) | 0);
			const scoreMode: ScoreMode = opts.score ?? "l2";

			const efSearch = Math.max(this.cfg.baseEfSearch, topK * 2);
			await this.wasm.setEfSearch(efSearch);

			let annK = Math.max(topK, topK * this.cfg.rerankMultiplier);
			annK = Math.min(annK, this.cfg.maxAnnK);
			if (this.wasmMaxEf > 0) annK = Math.min(annK, this.wasmMaxEf);
			await this.ensureResultsCapAtLeast(annK);

			const qF32s: Float32Array[] = new Array(queries.length);
			const qI8s: Int8Array[] = new Array(queries.length);

			for (let i = 0; i < queries.length; i++) {
				const qF32 = await this.resolveToF32(queries[i]);
				if (qF32.length !== this.cfg.dim) {
					throw new Error(
						`Query[${i}] dim mismatch. Expected ${this.cfg.dim}, got ${qF32.length}`,
					);
				}
				qF32s[i] = qF32;
				qI8s[i] = this.quantizeToI8(qF32);
			}

			const raws = await this.wasm.searchMany(qI8s, annK);

			const filter = opts.filter;
			const allowedSet =
				filter && typeof filter === "object" && typeof filter !== "function"
					? this.meta.filterInternalIdSet(filter)
					: null;
			const predicate =
				typeof filter === "function" ? (filter as (m: TMeta) => boolean) : null;

			const perQueryCandidates: number[][] = new Array(queries.length);
			const union = new Set<number>();

			for (let qi = 0; qi < queries.length; qi++) {
				const raw = raws[qi] || [];
				const candidates: number[] = [];

				for (const r of raw) {
					if (allowedSet && !allowedSet.has(r.id)) continue;
					const item = this.meta.getByInternalId(r.id);
					if (!item || item.deleted) continue;
					if (predicate && !predicate(item.metadata)) continue;
					candidates.push(r.id);
					union.add(r.id);
					if (candidates.length >= annK) break;
				}

				perQueryCandidates[qi] = candidates;
			}

			const allVecs = await this.readF32Vectors(Array.from(union));
			const out: SearchResult<TMeta>[][] = new Array(queries.length);

			for (let qi = 0; qi < queries.length; qi++) {
				const qF32 = qF32s[qi];
				const candidates = perQueryCandidates[qi];

				const reranked: { internalId: number; l2: number }[] = [];
				for (const internalId of candidates) {
					const v = allVecs.get(internalId);
					if (!v) continue;
					reranked.push({ internalId, l2: this.l2SqF32(qF32, v) });
				}

				reranked.sort((a, b) => a.l2 - b.l2 || a.internalId - b.internalId);

				const results: SearchResult<TMeta>[] = [];
				for (const r of reranked) {
					const item = this.meta.getByInternalId(r.internalId);
					if (!item || item.deleted) continue;
					results.push({
						id: item.external_id,
						score: this.toScore(r.l2, scoreMode),
						metadata: item.metadata as TMeta,
					});
					if (results.length >= topK) break;
				}
				out[qi] = results;
			}

			return out;
		});
	}

	/**
	 * Rebuild ANN index, optionally compact storage.
	 * 重建 ANN 索引，可选进行存储压缩。
	 *
	 * EN:
	 * - compact=true (default): rewrite vectors & metadata to remove deleted items and make IDs contiguous,
	 *   then rebuild the ANN index. This can reclaim disk space.
	 * - compact=false: only rebuild ANN index (skips deleted), without rewriting files or reordering IDs.
	 *
	 * CN:
	 * - compact=true（默认）：重写向量/元数据文件以移除已删除项，并把 internal_id 压缩为连续，
	 *   然后重建 ANN 索引（可回收磁盘空间）。
	 * - compact=false：仅重建 ANN 索引（跳过 deleted），不重写文件、不重排 ID。
	 */
	async rebuild(
		opts: RebuildOptions = {},
	): Promise<{ rebuilt: number; capacity: number; compact: boolean }> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			return this.rebuildLocked(opts);
		});
	}

	/**
	 * Save a snapshot (ANN index dump + metadata + state).
	 * 保存快照（ANN 索引 dump + 元数据 + 状态信息）。
	 *
	 * EN:
	 * Persists current state to disk so that future `load()` can restore quickly.
	 * If `filePath` is provided, dump will be written there; otherwise default dump path is used.
	 *
	 * CN:
	 * 将当前状态持久化，便于下次 `load()` 快速恢复。
	 * 若传入 `filePath` 则写到指定路径，否则使用默认 dump 路径。
	 */
	async save(filePath?: string): Promise<void> {
		return this.withLock(async () => this.saveLocked(filePath));
	}

	/**
	 * Load snapshot from disk (if exists), then replay operation log if needed.
	 * 从磁盘加载快照（若存在），并在需要时回放操作日志。
	 *
	 * EN:
	 * - If dump exists and is valid: loads ANN index.
	 * - If dump is missing/corrupt and autoRebuildOnLoad is enabled: rebuild index from metadata+vectors.
	 * - Replays oplog to keep ANN index consistent with recent inserts.
	 *
	 * CN:
	 * - dump 存在且可用：加载 ANN 索引；
	 * - dump 缺失/损坏且开启 autoRebuildOnLoad：从元数据+向量重建；
	 * - 回放 oplog，确保 ANN 与最近的写入保持一致。
	 */
	async load(filePath?: string): Promise<void> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();

			const fp = filePath || this.dumpPath;

			const tmp = `${fp}.tmp`;
			if (!fs.existsSync(fp) && fs.existsSync(tmp)) {
				await atomicReplace(tmp, fp).catch(() => {});
			}

			let loaded = false;

			if (fs.existsSync(fp)) {
				try {
					await this.wasm.load(fp);
					loaded = true;
				} catch {
					loaded = false;
				}
			}

			await this.ensureVectorStoreReady();

			if (this.cfg.preloadVectors && fs.existsSync(this.vecPath)) {
				this.vecCache = await fs.promises.readFile(this.vecPath);
			}

			if (!loaded) {
				if (!this.cfg.autoRebuildOnLoad) return;

				if (this.meta.getActiveCount() > 0) {
					await this.rebuildLocked({ persist: true, compact: false });
				}
				return;
			}

			await this.replayOplogLocked();
		});
	}

	/**
	 * Export database records as JSONL (one JSON per line).
	 * 以 JSONL 格式导出数据库（每行一个 JSON）。
	 *
	 * EN:
	 * - Can optionally include deleted items and/or raw vectors.
	 * - Useful for backup, migration, or debugging.
	 *
	 * CN:
	 * - 可选包含已删除项、可选包含原始向量。
	 * - 适用于备份、迁移或调试。
	 */
	async exportJSONL(
		filePath: string,
		opts: ExportJSONLOptions = {},
	): Promise<void> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			await this.ensureVectorStoreReady();

			const includeDeleted = opts.includeDeleted === true;
			const includeVectors = opts.includeVectors === true;
			const onProgress = opts.onProgress;

			const dim = this.cfg.dim;
			const bpv = this.bytesPerVec();
			const maxId = this.meta.getNextInternalId();

			const ws = fs.createWriteStream(filePath, { encoding: "utf8" });

			const block = 256;
			const buf = Buffer.allocUnsafe(block * bpv);

			let done = 0;
			const total = includeDeleted
				? this.meta.getTotalCount()
				: this.meta.getActiveCount();

			for (let start = 0; start < maxId; start += block) {
				const count = Math.min(block, maxId - start);
				const bytes = count * bpv;
				const pos = start * bpv;

				if (includeVectors) {
					if (
						this.cfg.preloadVectors &&
						this.vecCache &&
						pos + bytes <= this.vecCache.length
					) {
						this.vecCache.copy(buf, 0, pos, pos + bytes);
					} else {
						const r = await this.vecFd!.read(buf, 0, bytes, pos);
						if (r.bytesRead !== bytes) continue;
					}
				}

				const f32 = includeVectors
					? new Float32Array(buf.buffer, buf.byteOffset, bytes / 4)
					: null;

				for (let j = 0; j < count; j++) {
					const internalId = start + j;
					const item = this.meta.getByInternalId(internalId);
					if (!item) continue;
					if (!includeDeleted && item.deleted) continue;

					const rec: any = {
						id: item.external_id,
						internal_id: item.internal_id,
						deleted: item.deleted === true,
						metadata: item.metadata,
					};

					if (includeVectors && f32) {
						const base = j * dim;
						rec.vector = Array.from(f32.subarray(base, base + dim));
					}

					ws.write(JSON.stringify(rec) + "\n");
					done++;
					onProgress?.(done, total);
				}
			}

			await new Promise<void>((resolve, reject) => {
				ws.end(() => resolve());
				ws.on("error", reject);
			});
		});
	}

	/**
	 * Import JSONL records into the database.
	 * 从 JSONL 导入数据到数据库。
	 *
	 * EN:
	 * - Each line should be a JSON object with at least `id`.
	 * - If `vector` is provided (array), it will be inserted directly (no embedding).
	 * - Otherwise, `input` is treated as text/binary and will be embedded.
	 *
	 * CN:
	 * - 每行一个 JSON，至少包含 `id`；
	 * - 若提供 `vector`（数组），将直接写入（不再做 embedding）；
	 * - 否则按文本/二进制输入生成向量再写入。
	 */
	async importJSONL(
		filePath: string,
		opts: ImportJSONLOptions = {},
	): Promise<void> {
		const batchSize = Math.max(1, (opts.batchSize ?? 256) | 0);
		const onProgress = opts.onProgress;

		const rl = readline.createInterface({
			input: fs.createReadStream(filePath, { encoding: "utf8" }),
			crlfDelay: Infinity,
		});

		let batch: InsertItem<TMeta>[] = [];
		let done = 0;

		for await (const line of rl) {
			const s = line.trim();
			if (!s) continue;

			let obj: any;
			try {
				obj = JSON.parse(s);
			} catch {
				continue;
			}

			const id = String(obj.id ?? "");
			if (!id) continue;

			let input: any = obj.vector;
			if (Array.isArray(input)) input = new Float32Array(input);

			batch.push({ id, input, metadata: obj.metadata });

			if (batch.length >= batchSize) {
				await this.insertMany(batch);
				done += batch.length;
				onProgress?.(done, -1);
				batch = [];
			}
		}

		if (batch.length) {
			await this.insertMany(batch);
			done += batch.length;
			onProgress?.(done, -1);
		}
	}

	/**
	 * Get database statistics (lightweight, no IO-heavy scan).
	 * 获取数据库统计信息（轻量，不做重 IO 扫描）。
	 *
	 * EN:
	 * Useful for monitoring: counts, capacity, model/dim, and whether vectors are preloaded.
	 *
	 * CN:
	 * 便于监控：包含条目数量、容量、模型/维度、是否预加载向量等。
	 */
	getStats() {
		return {
			mode: this.cfg.mode,
			collection: this.cfg.collection || undefined,
			model: this.cfg.modelName,
			dim: this.cfg.dim,

			items: this.meta.getTotalCount(),
			deletedCount: this.meta.getDeletedCount(),
			activeCount: this.meta.getActiveCount(),

			storageDir: this.cfg.storageDir,
			capacity: this.capacity,
			preloadVectors: this.cfg.preloadVectors,
			wasmMaxEf: this.wasmMaxEf || undefined,
		};
	}

	/**
	 * Close database and release resources (file handles, caches).
	 * 关闭数据库并释放资源（文件句柄、缓存等）。
	 *
	 * EN:
	 * After close, the instance should not be used unless reopened.
	 *
	 * CN:
	 * close 后不应继续使用该实例（需要重新 open）。
	 */
	async close(): Promise<void> {
		return this.withLock(async () => {
			await this.meta.close();
			if (this.vecFd) {
				await this.vecFd.close();
				this.vecFd = null;
			}
			if (this.oplogFd) {
				await this.oplogFd.close();
				this.oplogFd = null;
			}
			this.vecCache = null;
			this.embCache.clear();
		});
	}

	private bytesPerVec(): number {
		return this.cfg.dim * 4;
	}

	private normalizeF32InPlace(v: Float32Array): Float32Array {
		let normSq = 0;
		for (let i = 0; i < v.length; i++) normSq += v[i] * v[i];
		if (normSq <= 0) return v;
		const inv = 1 / Math.sqrt(normSq);
		for (let i = 0; i < v.length; i++) v[i] *= inv;
		return v;
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

	private toScore(l2Sq: number, mode: ScoreMode): number {
		if (mode === "l2") return l2Sq;
		const cosine = 1 - l2Sq / 2; // unit vectors: ||a-b||^2 = 2-2cos
		if (mode === "cosine") return Math.max(-1, Math.min(1, cosine));
		const sim = 1 - l2Sq / 4; // (cos+1)/2
		return Math.max(0, Math.min(1, sim));
	}

	private touchEmbCache(key: string, v: Float32Array) {
		if (this.embCacheMax <= 0) return;
		if (this.embCache.has(key)) this.embCache.delete(key);
		this.embCache.set(key, v);
		while (this.embCache.size > this.embCacheMax) {
			const firstKey = this.embCache.keys().next().value;
			if (firstKey) this.embCache.delete(firstKey);
		}
	}

	private async ensureVectorStoreReady(): Promise<void> {
		if (this.vecFd) return;
		const exists = fs.existsSync(this.vecPath);
		this.vecFd = await fs.promises.open(this.vecPath, exists ? "r+" : "w+");
		if (this.cfg.preloadVectors && exists) {
			this.vecCache = await fs.promises.readFile(this.vecPath);
		}
	}

	private ensureCacheCapacity(minBytes: number) {
		if (!this.cfg.preloadVectors) return;
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
	) {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const bpv = this.bytesPerVec();
		const totalBytes = vectors.length * bpv;
		const pos = startInternalId * bpv;

		const buf = Buffer.allocUnsafe(totalBytes);
		for (let i = 0; i < vectors.length; i++) {
			const v = vectors[i];
			Buffer.from(v.buffer, v.byteOffset, v.byteLength).copy(
				buf,
				i * bpv,
				0,
				bpv,
			);
		}

		const { bytesWritten } = await fd.write(buf, 0, totalBytes, pos);
		if (bytesWritten !== totalBytes) {
			throw new Error(
				`Vector store bulk write failed. bytesWritten=${bytesWritten}, total=${totalBytes}`,
			);
		}

		if (this.cfg.preloadVectors) {
			this.ensureCacheCapacity(pos + totalBytes);
			this.vecCache!.set(buf, pos);
		}
	}

	private async writeF32Vector(internalId: number, v: Float32Array) {
		return this.writeF32VectorsContiguous(internalId, [v]);
	}

	private async readF32Vectors(
		internalIds: number[],
	): Promise<Map<number, Float32Array>> {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const dim = this.cfg.dim;
		const bpv = this.bytesPerVec();

		const out = new Map<number, Float32Array>();
		if (!internalIds.length) return out;

		if (this.cfg.preloadVectors && this.vecCache) {
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

	private ensureWithinCapacity(internalId: number) {
		if (internalId < 0 || internalId >= this.capacity) {
			throw new Error(
				`Database capacity exceeded: internalId=${internalId}, capacity=${this.capacity}. ` +
					`Fix: call db.rebuild({ capacity: ${this.capacity * 2} }) or open with larger capacity.`,
			);
		}
	}

	private ensureAllocWithinCapacity(start: number, n: number) {
		const endExclusive = start + n;
		if (start < 0 || n <= 0 || endExclusive > this.capacity) {
			throw new Error(
				`Database capacity overflow: need [${start}, ${endExclusive}) but capacity=${this.capacity}. ` +
					`Fix: call db.rebuild({ capacity: ${this.capacity * 2} }) or open with larger capacity.`,
			);
		}
	}

	private async ensureResultsCapAtLeast(n: number) {
		if (n <= 0) return;
		if (this.wasmMaxEf > 0) n = Math.min(n, this.wasmMaxEf);

		const cur = await this.wasm.getResultsCap();
		if (cur >= n) return;

		let next = cur > 0 ? cur : 1000;
		while (next < n) next *= 2;
		await this.wasm.setResultsCap(next);
	}

	private async ensureOplogReady() {
		if (this.oplogFd) return;
		this.oplogFd = await fs.promises.open(this.oplogPath, "a");
	}

	private async appendOplogLocked(lines: string) {
		if (!lines) return;
		await this.ensureOplogReady();
		await this.oplogFd!.write(lines, null, "utf8");
		await this.oplogFd!.sync();
	}

	private async maybeAutoRebuildLocked() {
		const thr = this.cfg.deletedRebuildThreshold;
		if (!(thr > 0)) return;

		const total = this.meta.getTotalCount();
		if (total <= 0) return;

		const ratio = this.deletedSinceRebuild / total;
		if (ratio < thr) return;

		await this.rebuildLocked({ persist: true, compact: false });
	}

	private async resolveToF32(
		input: InsertItem["input"],
	): Promise<Float32Array> {
		if (typeof input === "string") {
			if (this.embCacheMax > 0) {
				const cached = this.embCache.get(input);
				if (cached) {
					this.embCache.delete(input);
					this.embCache.set(input, cached);
					return new Float32Array(cached);
				}
			}

			const out = await this.embedder.embed(input);
			const v = this.normalizeF32InPlace(new Float32Array(out));
			this.touchEmbCache(input, new Float32Array(v));
			return v;
		}

		if (input instanceof Buffer || input instanceof Uint8Array) {
			const out = await this.embedder.embed(input);
			return this.normalizeF32InPlace(new Float32Array(out));
		}

		if (input instanceof Float32Array)
			return this.normalizeF32InPlace(new Float32Array(input));
		return this.normalizeF32InPlace(new Float32Array(input));
	}

	/**
	 * compact=false: only rebuild HNSW index (skip deleted), no file rewrite, no ID reorder.
	 * compact=true : true compaction (rewrite vectors/meta, reorder internal_id contiguous) + rebuild HNSW.
	 */
	private async rebuildLocked(
		opts: RebuildOptions,
	): Promise<{ rebuilt: number; capacity: number; compact: boolean }> {
		const persist = opts.persist !== false;
		const onProgress = opts.onProgress;
		const compact = opts.compact !== false; // ✅ default true

		if (compact) {
			const r = await this.compactRebuildLocked(opts);
			if (persist) await this.saveLocked();
			return { ...r, compact: true };
		}

		// --- old behavior: HNSW rebuild only ---
		if (opts.capacity && opts.capacity > this.capacity) {
			this.capacity = opts.capacity | 0;
		}

		// must be >= nextInternalId (otherwise existing internal IDs out of range)
		const need = this.meta.getNextInternalId();
		if (this.capacity < need) {
			throw new Error(
				`rebuild capacity too small: capacity=${this.capacity}, need>=${need}`,
			);
		}

		await this.wasm.reinitIndex(this.capacity);

		await this.ensureVectorStoreReady();
		const dim = this.cfg.dim;
		const bpv = this.bytesPerVec();
		const maxId = this.meta.getNextInternalId();
		const totalActive = this.meta.getActiveCount();

		const block = 256; // tuned for memory/GC balance
		const buf = Buffer.allocUnsafe(block * bpv);

		// reuse packed buffers
		const ids = new Int32Array(block);
		const packed = new Int8Array(block * dim);

		let done = 0;

		for (let start = 0; start < maxId; start += block) {
			const count = Math.min(block, maxId - start);
			const bytes = count * bpv;
			const pos = start * bpv;

			let bytesRead = 0;

			if (this.cfg.preloadVectors && this.vecCache) {
				if (pos + bytes <= this.vecCache.length) {
					this.vecCache.copy(buf, 0, pos, pos + bytes);
					bytesRead = bytes;
				}
			}

			if (bytesRead === 0) {
				const r = await this.vecFd!.read(buf, 0, bytes, pos);
				bytesRead = r.bytesRead;
			}

			const f32 = new Float32Array(buf.buffer, buf.byteOffset, bytesRead / 4);

			let w = 0;
			for (let j = 0; j < count; j++) {
				const internalId = start + j;
				const item = this.meta.getByInternalId(internalId);
				if (!item || item.deleted) continue;

				ids[w] = internalId;

				const base = j * dim;
				const outBase = w * dim;
				for (let k = 0; k < dim; k++) {
					let x = f32[base + k];
					if (x > 1) x = 1;
					else if (x < -1) x = -1;
					let q = Math.round(x * 127);
					if (q > 127) q = 127;
					else if (q < -127) q = -127;
					packed[outBase + k] = q;
				}

				w++;
			}

			if (w > 0) {
				await this.wasm.insertManyPacked(
					ids.subarray(0, w),
					packed.subarray(0, w * dim),
					dim,
				);
				done += w;
				onProgress?.(done, totalActive);
			}
		}

		this.deletedSinceRebuild = 0;

		if (persist) await this.saveLocked();

		return { rebuilt: done, capacity: this.capacity, compact: false };
	}

	/**
	 * ✅ True compaction:
	 * - rewrite vectors.f32.bin (only alive, contiguous new internal_id)
	 * - rewrite metadata.json (no deleted, new internal_id)
	 * - reset capacity >= active and (opts.capacity if larger)
	 * - rebuild HNSW based on new IDs
	 */
	private async compactRebuildLocked(
		opts: RebuildOptions,
	): Promise<{ rebuilt: number; capacity: number }> {
		const onProgress = opts.onProgress;

		if (!this.isReady) await this.init();
		await this.ensureVectorStoreReady();

		// 1) snapshot alive items
		const all = this.meta.allItems();
		const alive = all
			.filter((x) => x && x.deleted !== true)
			.sort((a, b) => a.internal_id - b.internal_id);

		const totalActive = alive.length;

		// new capacity: at least active, plus user requested (if any)
		let nextCap = Math.max(1, totalActive);
		if (
			opts.capacity &&
			Number.isFinite(opts.capacity) &&
			opts.capacity > nextCap
		) {
			nextCap = opts.capacity | 0;
		}
		this.capacity = nextCap;

		// 2) build old->new id mapping (contiguous)
		const idMap = new Map<number, number>();
		for (let i = 0; i < alive.length; i++) idMap.set(alive[i].internal_id, i);

		// 3) rewrite vectors to tmp
		const bpv = this.bytesPerVec();
		const vecTmp = makeUniqueTmpPath(this.vecPath);
		const metaTmp = makeUniqueTmpPath(this.metaPath);

		// cleanup leftovers
		await fs.promises.unlink(vecTmp).catch(() => {});
		await fs.promises.unlink(metaTmp).catch(() => {});

		const outVecFd = await fs.promises.open(vecTmp, "w+");

		const BATCH = 512;
		let done = 0;

		for (let i = 0; i < alive.length; i += BATCH) {
			const chunk = alive.slice(i, i + BATCH);
			const oldIds = chunk.map((x) => x.internal_id);

			const vecMap = await this.readF32Vectors(oldIds);

			const buf = Buffer.allocUnsafe(chunk.length * bpv);

			for (let j = 0; j < chunk.length; j++) {
				const oldId = chunk[j].internal_id;
				const v = vecMap.get(oldId);
				if (!v) {
					buf.fill(0, j * bpv, (j + 1) * bpv);
					continue;
				}
				Buffer.from(v.buffer, v.byteOffset, v.byteLength).copy(
					buf,
					j * bpv,
					0,
					bpv,
				);
			}

			const pos = i * bpv; // newId starts from 0
			const w = await outVecFd.write(buf, 0, buf.length, pos);
			if (w.bytesWritten !== buf.length) {
				await outVecFd.close().catch(() => {});
				throw new Error(
					`compact vectors short write: written=${w.bytesWritten}, need=${buf.length}`,
				);
			}

			done += chunk.length;
			onProgress?.(done, totalActive);
		}

		await outVecFd.sync().catch(() => {});
		await outVecFd.close();

		// 4) rewrite metadata to tmp (new MetaDB)
		{
			const tmpMeta = new MetaDB(metaTmp);
			await tmpMeta.ready();
			tmpMeta.beginBulk();

			let commit = false;
			try {
				const entries = alive.map((it) => ({
					external_id: it.external_id,
					internal_id: idMap.get(it.internal_id)!,
					metadata: it.metadata,
					deleted: false,
				}));
				tmpMeta.addMany(entries);
				commit = true;
			} finally {
				await tmpMeta.endBulk(commit);
				await tmpMeta.close();
			}
		}

		// 5) swap files atomically
		// close old handles first (Windows safety)
		if (this.vecFd) {
			await this.vecFd.close().catch(() => {});
			this.vecFd = null;
		}
		this.vecCache = null;

		await atomicReplace(vecTmp, this.vecPath);
		await atomicReplace(metaTmp, this.metaPath);

		// reopen meta/vector store
		await this.meta.close().catch(() => {});
		this.meta = new MetaDB(this.metaPath);
		await this.meta.ready();

		await this.ensureVectorStoreReady();
		if (this.cfg.preloadVectors && fs.existsSync(this.vecPath)) {
			this.vecCache = await fs.promises.readFile(this.vecPath);
		}

		// after compaction: there are no deletes
		this.deletedSinceRebuild = 0;

		// 6) rebuild HNSW from new contiguous store
		await this.wasm.reinitIndex(this.capacity);

		const dim = this.cfg.dim;

		const block = 256;
		const buf = Buffer.allocUnsafe(block * bpv);
		const ids = new Int32Array(block);
		const packed = new Int8Array(block * dim);

		let rebuilt = 0;

		for (let start = 0; start < totalActive; start += block) {
			const count = Math.min(block, totalActive - start);
			const bytes = count * bpv;
			const pos = start * bpv;

			let bytesRead = 0;
			if (this.cfg.preloadVectors && this.vecCache) {
				if (pos + bytes <= this.vecCache.length) {
					this.vecCache.copy(buf, 0, pos, pos + bytes);
					bytesRead = bytes;
				}
			}
			if (bytesRead === 0) {
				const r = await this.vecFd!.read(buf, 0, bytes, pos);
				bytesRead = r.bytesRead;
			}

			if (bytesRead !== bytes) {
				throw new Error(
					`compact rebuild short read: pos=${pos} need=${bytes} got=${bytesRead}`,
				);
			}

			const f32 = new Float32Array(buf.buffer, buf.byteOffset, bytesRead / 4);

			let w = 0;
			for (let j = 0; j < count; j++) {
				const internalId = start + j; // new id
				ids[w] = internalId;

				const base = j * dim;
				const outBase = w * dim;

				for (let k = 0; k < dim; k++) {
					let x = f32[base + k];
					if (x > 1) x = 1;
					else if (x < -1) x = -1;
					let q = Math.round(x * 127);
					if (q > 127) q = 127;
					else if (q < -127) q = -127;
					packed[outBase + k] = q;
				}

				w++;
			}

			if (w > 0) {
				await this.wasm.insertManyPacked(
					ids.subarray(0, w),
					packed.subarray(0, w * dim),
					dim,
				);
				rebuilt += w;
				onProgress?.(rebuilt, totalActive);
			}
		}

		return { rebuilt, capacity: this.capacity };
	}

	private async replayOplogLocked(): Promise<void> {
		await this.ensureOplogReady();
		const stat = await fs.promises.stat(this.oplogPath).catch(() => null);
		if (!stat || stat.size <= 0) return;

		const text = await fs.promises
			.readFile(this.oplogPath, "utf8")
			.catch(() => "");
		if (!text) return;

		const up = new Set<number>();
		const lines = text.split(/\r?\n/);
		for (const line of lines) {
			const s = line.trim();
			if (!s) continue;
			const [op, idStr] = s.split(/\s+/);
			const id = Number(idStr);
			if (!Number.isInteger(id) || id < 0) continue;
			if (op === "U") up.add(id);
			// "D" is metadata-only today; kept for future hard-delete in WASM
		}

		const ids = Array.from(up).sort((a, b) => a - b);
		if (!ids.length) return;

		const existing: number[] = [];
		for (const id of ids) {
			const item = this.meta.getByInternalId(id);
			if (!item || item.deleted) continue;
			this.ensureWithinCapacity(id);
			existing.push(id);
		}
		if (!existing.length) return;

		const vecMap = await this.readF32Vectors(existing);

		const pairs: { id: number; vectorI8: Int8Array }[] = [];
		for (const id of existing) {
			const v = vecMap.get(id);
			if (!v) continue;
			pairs.push({ id, vectorI8: this.quantizeToI8(v) });
		}
		if (pairs.length) await this.wasm.insertMany(pairs);
	}

	private async saveLocked(filePath?: string): Promise<void> {
		if (!this.isReady) await this.init();

		await this.meta.saveNow();
		if (this.vecFd) await this.vecFd.sync();

		const finalDump = filePath || this.dumpPath;
		const tmpDump = makeUniqueTmpPath(finalDump);

		// write dump tmp then atomic replace
		await this.wasm.save(tmpDump);
		await atomicReplace(tmpDump, finalDump);

		// write state tmp then atomic replace
		const state = {
			savedAt: new Date().toISOString(),
			collection: this.cfg.collection,
			model: this.cfg.modelName,
			dim: this.cfg.dim,
			capacity: this.capacity,
			totalCount: this.meta.getTotalCount(),
			deletedCount: this.meta.getDeletedCount(),
			nextInternalId: this.meta.getNextInternalId(),
		};

		const tmpState = makeUniqueTmpPath(this.statePath);
		await fs.promises.writeFile(tmpState, JSON.stringify(state, null, 2));
		await atomicReplace(tmpState, this.statePath);

		// ✅ truncate oplog after dump snapshot (Windows-safe)
		try {
			// Close first: Windows + a+ fd + ftruncate is often EPERM
			if (this.oplogFd) {
				await this.oplogFd.close().catch(() => {});
				this.oplogFd = null;
			}

			// Use path-based truncate (works reliably on Windows)
			await fs.promises.truncate(this.oplogPath, 0).catch(() => {});
		} finally {
			// Reopen lazily when needed (or reopen now if you prefer)
			// await this.ensureOplogReady();
		}
	}
}
