// src/index.ts
import { WasmBridge } from "./core/wasm-bridge";
import { MetaDB } from "./storage/meta-db";
import { LocalEmbedder } from "./embedder";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";

// ✅ 读取项目根目录的 .env（你也可以不配 .env，所有参数都能在 open() 里传）
dotenv.config({ path: path.join(__dirname, "../.env") });

/**
 * 三档预设：
 * - fast：更快、更省内存（召回更少）
 * - balanced：默认档，兼顾速度和效果
 * - accurate：更准（通常更慢）
 */
export type ModePreset = "fast" | "balanced" | "accurate";

/**
 * 搜索结果：
 * - score 越小越相似（L2 距离平方）
 */
export interface SearchResult<TMeta = any> {
	id: string;
	score: number;
	metadata: TMeta;
}

/**
 * 插入条目：
 * - input 既可以是向量，也可以是文本/图片（会自动 embed）
 */
export interface InsertItem<TMeta = any> {
	id: string;
	input: number[] | Float32Array | string | Buffer | Uint8Array;
	metadata?: TMeta;
}

/**
 * 搜索选项：
 * - topK：要返回多少条
 * - filter：可以用 Loki 的 query object，也可以用 predicate(metadata)=>boolean
 */
export interface SearchOptions<TMeta = any> {
	topK?: number;
	filter?: any | ((metadata: TMeta) => boolean);
}

/**
 * 打开数据库时的配置：
 * 你只需要关心少量字段（storageDir/modelName/mode），其他都可以不填。
 */
export interface DBOpenOptions {
	/**
	 * 数据目录（默认 ./data）
	 * 会生成/使用：
	 * - metadata.json
	 * - vectors.f32.bin
	 * - dump.bin
	 */
	storageDir?: string;

	/**
	 * 用于自动 embedding 的模型名
	 * - text 默认：Xenova/all-MiniLM-L6-v2（384 维）
	 * - clip 示例：Xenova/clip-vit-base-patch32（512 维）
	 */
	modelName?: string;
	modelArchitecture?: "text" | "clip";

	/**
	 * 一键档位：fast / balanced / accurate
	 */
	mode?: ModePreset;

	/**
	 * 高级项（可不填，系统会自动推导）
	 */
	dim?: number;
	capacity?: number;
	preloadVectors?: boolean;
	seed?: number;

	/**
	 * HNSW 高级参数（不懂就别填）
	 * 注意：这些参数改变后通常需要重建索引才会完全生效
	 */
	m?: number;
	ef_construction?: number;
}

/**
 * 内部最终配置：对外不暴露（避免吓到用户）
 */
type InternalResolvedConfig = Required<
	Pick<DBOpenOptions, "storageDir" | "modelName" | "mode" | "preloadVectors">
> & {
	modelArchitecture: "text" | "clip";
	dim: number;
	capacity: number;
	seed?: number;

	// HNSW build 参数
	m: number;
	ef_construction: number;

	// 查询策略（运行时可变；用户一般不需要理解）
	baseEfSearch: number;
	rerankMultiplier: number;
	maxAnnK: number;
	resultsCap: number;
};

/**
 * ✅ 根据模型大致推导维度：
 * - text -> 384
 * - clip -> 512
 *
 * 你也可以显式传 dim 覆盖它。
 */
function inferDimFromModel(modelName: string, arch: "text" | "clip"): number {
	return arch === "clip" ? 512 : 384;
}

/**
 * ✅ 推导模型架构：
 * - 你显式传 arch 就用你的
 * - 否则：modelName 里包含 clip 就当 clip，否则当 text
 */
function resolveArch(
	modelName?: string,
	arch?: "text" | "clip",
): "text" | "clip" {
	if (arch) return arch;
	if (!modelName) return "text";
	return modelName.toLowerCase().includes("clip") ? "clip" : "text";
}

/**
 * ✅ 三档预设参数（给不想研究 HNSW 的人用）
 *
 * 说明：
 * - m / ef_construction 影响建库质量与速度（一般需要重建才能发挥完整效果）
 * - baseEfSearch / rerankMultiplier 影响查询速度与准确率（运行时可动态调）
 *
 * ⚠️ WASM 侧 MAX_EF=4096，因此 maxAnnK/resultsCap 不建议超过 4096
 */
function resolvePreset(mode: ModePreset) {
	switch (mode) {
		case "fast":
			return {
				m: 12,
				ef_construction: 150,

				baseEfSearch: 80,
				rerankMultiplier: 40,

				maxAnnK: 2048,
				resultsCap: 2048,
			};

		case "accurate":
			return {
				m: 24,
				ef_construction: 600,

				// 经验值：不要把默认候选池怼太大，否则延迟会明显上升
				baseEfSearch: 200,
				rerankMultiplier: 250,

				maxAnnK: 4096,
				resultsCap: 4096,
			};

		case "balanced":
		default:
			return {
				m: 16,
				ef_construction: 300,

				baseEfSearch: 140,
				rerankMultiplier: 120,

				maxAnnK: 4096,
				resultsCap: 4096,
			};
	}
}

/**
 * ✅ 把“用户输入的 opts + .env”整合成最终配置
 * 优先级：opts > env > 默认值
 */
function resolveOpenConfig(opts: DBOpenOptions): InternalResolvedConfig {
	const storageDir =
		opts.storageDir ||
		process.env.MINIVECTOR_STORAGE_DIR ||
		path.join(process.cwd(), "data");

	const modelName =
		opts.modelName || process.env.MODEL_NAME || "Xenova/all-MiniLM-L6-v2";

	const modelArchitecture = resolveArch(modelName, opts.modelArchitecture);

	const dim =
		opts.dim ||
		Number(process.env.VECTOR_DIM ?? 0) ||
		inferDimFromModel(modelName, modelArchitecture);

	const mode: ModePreset =
		opts.mode || (process.env.MINIVECTOR_MODE as ModePreset) || "balanced";
	const preset = resolvePreset(mode);

	const capacity =
		opts.capacity || Number(process.env.HNSW_CAPACITY ?? 0) || 1_200_000;

	const preloadVectors = !!(
		opts.preloadVectors ?? process.env.PRELOAD_VECTORS === "1"
	);

	return {
		storageDir,
		modelName,
		mode,
		preloadVectors,
		modelArchitecture,

		dim,
		capacity,
		seed: opts.seed,

		m: opts.m || Number(process.env.HNSW_M ?? 0) || preset.m,
		ef_construction:
			opts.ef_construction ||
			Number(process.env.HNSW_EF ?? 0) ||
			preset.ef_construction,

		baseEfSearch:
			Number(process.env.BASE_EF_SEARCH ?? 0) || preset.baseEfSearch,
		rerankMultiplier:
			Number(process.env.RERANK_MULTIPLIER ?? 0) || preset.rerankMultiplier,
		maxAnnK: Number(process.env.MAX_ANN_K ?? 0) || preset.maxAnnK,
		resultsCap: Number(process.env.HNSW_RESULTS_CAP ?? 0) || preset.resultsCap,
	};
}

/**
 * MiniVectorDB（给不想研究 HNSW 参数的人）
 *
 * 你通常只需要：
 * 1) const db = await MiniVectorDB.open({ storageDir, mode })
 * 2) await db.insert({ id, input, metadata })
 * 3) const results = await db.search(query, { topK })
 * 4) await db.save()
 */
export class MiniVectorDB<TMeta = any> {
	// =========================
	// ✅ 用户最关心的：配置与入口
	// =========================

	/** 打开时最终生效的配置（可用于 debug） */
	public readonly cfg: InternalResolvedConfig;

	/**
	 * ✅ 一行打开数据库（推荐用这个）
	 * - 会自动 init()
	 * - 会尝试 load()（如果 storageDir 里有 dump.bin）
	 */
	static async open<TMeta = any>(
		opts: DBOpenOptions = {},
	): Promise<MiniVectorDB<TMeta>> {
		const cfg = resolveOpenConfig(opts);
		const db = new MiniVectorDB<TMeta>(cfg);
		await db.init();

		// 有 dump 就加载，没有就当空库
		try {
			await db.load();
		} catch {
			// ignore
		}

		return db;
	}

	/**
	 * ✅ 插入一条（input 可以是：向量 / 文本 / 图片 buffer）
	 */
	async insert(item: InsertItem<TMeta>): Promise<void> {
		await this.insertMany([item]);
	}

	/**
	 * ✅ 批量插入
	 * - 新数据会分配 internal_id
	 * - 向量会写入 vectors.f32.bin
	 * - ANN（WASM/HNSW）会插入/更新 int8 量化向量
	 */
	async insertMany(items: InsertItem<TMeta>[]): Promise<void> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			if (!items.length) return;

			const expectedDim = this.cfg.dim;

			const externalIds = items.map((it) => it.id);
			const existingMap = this.meta.getMany(externalIds);

			this.meta.beginBulk();
			let commit = false;

			try {
				// 1) 为新数据预分配 internal_id（确保连续、便于批量写向量）
				let newItemsCount = 0;
				for (const it of items) if (!existingMap.get(it.id)) newItemsCount++;

				let newStartId = 0;
				if (newItemsCount > 0) {
					newStartId = this.meta.allocInternalIds(newItemsCount);
					this.ensureAllocWithinCapacity(newStartId, newItemsCount);
				}

				// 2) 准备：向量、量化向量、internal_id、元数据
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

				for (let i = 0; i < items.length; i++) {
					const { id, input, metadata } = items[i];
					const f32 = await this.resolveToF32(input); // 自动 embed + normalize

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
							metadata: metadata ?? ({} as any),
						};
					}

					f32s[i] = f32;
					q8s[i] = q8;
				}

				// 3) 写入 float32 向量到磁盘（新数据尽量按连续块写，减少 IO）
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

				// 更新已有数据（随机写）
				for (let i = 0; i < items.length; i++) {
					if (!isNew[i]) await this.writeF32Vector(internalIds[i], f32s[i]);
				}

				if (this.vecFd) await this.vecFd.sync();

				// 4) 更新 ANN（WASM/HNSW）里的 int8 向量
				for (let i = 0; i < items.length; i++) {
					const internalId = internalIds[i];
					const q8 = q8s[i];

					if (isNew[i]) {
						await this.wasm.insert(internalId, q8);
					} else {
						if (await this.wasm.hasNode(internalId))
							await this.wasm.updateVector(internalId, q8);
						else await this.wasm.insert(internalId, q8);
					}
				}

				// 5) 最后提交元数据（失败就回滚）
				this.meta.addMany(metaEntries, existingMap);
				commit = true;
			} finally {
				await this.meta.endBulk(commit);
			}
		});
	}

	/**
	 * ✅ 搜索（query 可以是：向量 / 文本 / 图片 buffer）
	 *
	 * 返回结果：
	 * - id：你插入时给的 external id
	 * - score：越小越相似（L2 距离平方）
	 */
	async search(
		query: InsertItem["input"],
		opts: SearchOptions<TMeta> = {},
	): Promise<SearchResult<TMeta>[]> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();

			const topK = Math.max(1, (opts.topK ?? 10) | 0);

			// 1) 把 query 转成 float32 向量（自动 embed + normalize）
			const qF32 = await this.resolveToF32(query);
			if (qF32.length !== this.cfg.dim) {
				throw new Error(
					`Query vector dimension mismatch. Expected ${this.cfg.dim}, got ${qF32.length}`,
				);
			}

			// 2) 量化成 int8，先用 HNSW 召回候选
			const qI8 = this.quantizeToI8(qF32);

			// efSearch：至少 topK*2，同时不低于 preset 的 base
			const efSearch = Math.max(this.cfg.baseEfSearch, topK * 2);
			await this.wasm.setEfSearch(efSearch);

			// annK：候选池大小 = topK * rerankMultiplier（并受 wasm/MAX_EF 限制）
			const mult = this.cfg.rerankMultiplier;
			let annK = Math.max(topK, topK * mult);
			annK = Math.min(annK, this.cfg.maxAnnK);
			if (this.wasmMaxEf > 0) annK = Math.min(annK, this.wasmMaxEf);

			// WASM 侧 results buffer 不够会扩容
			await this.ensureResultsCapAtLeast(annK);

			const raw = await this.wasm.search(qI8, annK);
			if (!raw.length) return [];

			// 3) 可选过滤（metadata filter）
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
				if (!item) continue;

				if (predicate && !predicate(item.metadata)) continue;

				candidates.push(r.id);
				if (candidates.length >= annK) break;
			}
			if (!candidates.length) return [];

			// 4) 精排：用磁盘里的 float32 原始向量算 L2（更准）
			const vecMap = await this.readF32Vectors(candidates);
			const reranked: { internalId: number; score: number }[] = [];

			for (const internalId of candidates) {
				const v = vecMap.get(internalId);
				if (!v) continue;
				reranked.push({ internalId, score: this.l2SqF32(qF32, v) });
			}

			// ✅ 重要：score 相等时按 internalId 稳定排序（避免测试/线上随机抖动）
			reranked.sort((a, b) => a.score - b.score || a.internalId - b.internalId);

			const results: SearchResult<TMeta>[] = [];
			for (const r of reranked) {
				const item = this.meta.getByInternalId(r.internalId);
				if (!item) continue;

				results.push({
					id: item.external_id,
					score: r.score,
					metadata: item.metadata as TMeta,
				});

				if (results.length >= topK) break;
			}

			return results;
		});
	}

	/**
	 * ✅ 保存：
	 * - metadata.json（Loki）
	 * - dump.bin（HNSW + 量化向量）
	 * - vectors.f32.bin（float32 原始向量文件会 sync）
	 */
	async save(filePath?: string): Promise<void> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();
			await this.meta.saveNow();
			await this.wasm.save(filePath || this.dumpPath);
			if (this.vecFd) await this.vecFd.sync();
		});
	}

	/**
	 * ✅ 加载（默认 storageDir/dump.bin）
	 * 注意：这只加载 ANN 结构；metadata/vector 文件仍在 storageDir 下
	 */
	async load(filePath?: string): Promise<void> {
		return this.withLock(async () => {
			if (!this.isReady) await this.init();

			const fp = filePath || this.dumpPath;
			if (!fs.existsSync(fp)) return;

			await this.wasm.load(fp);
			await this.ensureVectorStoreReady();

			// preloadVectors=1 时会把 vectors.f32.bin 整个读进内存（更快，但更吃内存）
			if (this.cfg.preloadVectors && fs.existsSync(this.vecPath)) {
				this.vecCache = await fs.promises.readFile(this.vecPath);
			}
		});
	}

	/**
	 * ✅ 返回一些运行信息（用于 debug / 看当前配置）
	 */
	getStats() {
		return {
			mode: this.cfg.mode,
			model: this.cfg.modelName,
			dim: this.cfg.dim,
			items: this.meta.items?.count?.() ?? 0,
			storageDir: this.cfg.storageDir,
			capacity: this.cfg.capacity,
			preloadVectors: this.cfg.preloadVectors,
			wasmMaxEf: this.wasmMaxEf || undefined,
		};
	}

	/**
	 * ✅ 关闭数据库（释放文件句柄/缓存）
	 */
	async close(): Promise<void> {
		return this.withLock(async () => {
			await this.meta.close();
			if (this.vecFd) {
				await this.vecFd.close();
				this.vecFd = null;
			}
			this.vecCache = null;
		});
	}

	// =========================
	// 内部：构造/init/资源管理
	// =========================

	private wasm = new WasmBridge();
	private meta: MetaDB;
	private embedder: LocalEmbedder;

	private isReady = false;

	private vecPath: string;
	private dumpPath: string;
	private metaPath: string;

	private vecFd: fs.promises.FileHandle | null = null;
	private vecCache: Buffer | null = null;

	private wasmMaxEf = 0;

	private constructor(cfg: InternalResolvedConfig) {
		this.cfg = cfg;

		if (!fs.existsSync(cfg.storageDir))
			fs.mkdirSync(cfg.storageDir, { recursive: true });

		this.metaPath = path.join(cfg.storageDir, "metadata.json");
		this.vecPath = path.join(cfg.storageDir, "vectors.f32.bin");
		this.dumpPath = path.join(cfg.storageDir, "dump.bin");

		this.meta = new MetaDB(this.metaPath);
		this.embedder = new LocalEmbedder(cfg.modelName, cfg.modelArchitecture);
	}

	/**
	 * 初始化底层资源（WASM + MetaDB + vector store）
	 * - open() 会自动调用它
	 */
	async init(): Promise<void> {
		return this.withLock(async () => {
			if (this.isReady) return;

			await this.wasm.init({
				dim: this.cfg.dim,
				m: this.cfg.m,
				ef: this.cfg.ef_construction,
				efSearch: this.cfg.baseEfSearch,
				capacity: this.cfg.capacity,
				seed: this.cfg.seed,
				resultsCap: this.cfg.resultsCap,
			});

			this.wasmMaxEf = await this.wasm.getMaxEf();

			await this.meta.ready();
			await this.ensureVectorStoreReady();

			this.isReady = true;
		});
	}

	// =========================
	// 内部：并发保护（避免同时读写）
	// =========================

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

	// =========================
	// 内部：向量工具/IO
	// =========================

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

	private async ensureVectorStoreReady(): Promise<void> {
		if (this.vecFd) return;

		const exists = fs.existsSync(this.vecPath);
		this.vecFd = await fs.promises.open(this.vecPath, exists ? "r+" : "w+");

		if (this.cfg.preloadVectors && exists) {
			this.vecCache = await fs.promises.readFile(this.vecPath);
		}
	}

	private async resolveToF32(
		input: InsertItem["input"],
	): Promise<Float32Array> {
		let v: Float32Array;

		// 文本/图片：走 embedder
		if (
			typeof input === "string" ||
			input instanceof Buffer ||
			input instanceof Uint8Array
		) {
			v = await this.embedder.embed(input);
		} else if (input instanceof Float32Array) {
			// Float32Array：复制一份避免外部修改
			v = new Float32Array(input);
		} else {
			// number[]：转成 Float32Array
			v = new Float32Array(input);
		}

		return this.normalizeF32InPlace(v);
	}

	// =========================
	// 内部：向量文件写入（Float32 原始向量）
	// =========================

	/**
	 * preloadVectors=1 时：需要确保内存缓存至少能覆盖到 minBytes
	 * - 缓存不够会扩容（并用 0 填充新增部分）
	 * - preloadVectors=0 时不会启用此逻辑
	 */
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

	/**
	 * 把一段“连续 internalId”的向量一次性写入磁盘（更高效）
	 *
	 * startInternalId：第一条向量对应的 internalId
	 * vectors：连续向量数组（vectors[0] 写到 startInternalId）
	 *
	 * 同时：
	 * - preloadVectors=1 会同步写入内存缓存（让后续精排读更快）
	 */
	private async writeF32VectorsContiguous(
		startInternalId: number,
		vectors: Float32Array[],
	): Promise<void> {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const bpv = this.bytesPerVec();

		const totalBytes = vectors.length * bpv;
		const pos = startInternalId * bpv;

		// ✅ 把多个 Float32Array 打包成一个 Buffer，一次写入
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

		// ✅ 如果启用了 preloadVectors，把写入内容同步到内存缓存
		if (this.cfg.preloadVectors) {
			this.ensureCacheCapacity(pos + totalBytes);
			this.vecCache!.set(buf, pos);
		}
	}

	/**
	 * 写入单条向量（实际上复用 contiguous 写入）
	 */
	private async writeF32Vector(
		internalId: number,
		v: Float32Array,
	): Promise<void> {
		await this.writeF32VectorsContiguous(internalId, [v]);
	}

	// =========================
	// 内部：向量文件读取（用于精排 rerank）
	// =========================

	/**
	 * 读取若干 internalId 对应的 float32 向量（用于精排）
	 *
	 * 读取策略：
	 * - preloadVectors=1：直接从内存缓存取（最快）
	 * - 否则：对 internalIds 去重排序，然后尽量按“连续区间”批量读取，减少随机 IO
	 */
	private async readF32Vectors(
		internalIds: number[],
	): Promise<Map<number, Float32Array>> {
		await this.ensureVectorStoreReady();
		const fd = this.vecFd!;
		const dim = this.cfg.dim;
		const bpv = this.bytesPerVec();

		const out = new Map<number, Float32Array>();
		if (!internalIds.length) return out;

		// ✅ 内存缓存路径：直接从 vecCache 读
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

		// ✅ 磁盘读取路径：去重 + 排序，按连续区间批量读
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

			// 拆回单条向量
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

	// =========================
	// 内部：容量保护（给用户更可读的错误）
	// =========================

	/**
	 * 检查某个 internalId 是否越界
	 * - 如果越界，告诉用户应该增大 capacity
	 */
	private ensureWithinCapacity(internalId: number) {
		const cap = this.cfg.capacity;
		if (internalId < 0 || internalId >= cap) {
			throw new Error(
				`Database capacity exceeded: internalId=${internalId}, capacity=${cap}. ` +
					`Fix: increase capacity when opening DB, e.g. MiniVectorDB.open({ capacity: ${cap * 2} }).`,
			);
		}
	}

	/**
	 * 检查一段 [start, start+n) 是否越界（用于批量预分配 internalId）
	 */
	private ensureAllocWithinCapacity(start: number, n: number) {
		const cap = this.cfg.capacity;
		const endExclusive = start + n;

		if (start < 0 || n <= 0 || endExclusive > cap) {
			throw new Error(
				`Database capacity overflow: need [${start}, ${endExclusive}) but capacity=${cap}. ` +
					`Fix: open DB with larger capacity, e.g. MiniVectorDB.open({ capacity: ${cap * 2} }).`,
			);
		}
	}

	// =========================
	// 内部：WASM 结果缓冲区管理（避免候选数变大时溢出）
	// =========================

	/**
	 * 确保 WASM 侧 resultsCap >= n
	 *
	 * 说明：
	 * - annK 变大时，WASM 需要更大的结果缓冲区
	 * - 这里用“指数扩容”（next *= 2）减少频繁 setResultsCap
	 * - 同时受 wasmMaxEf 限制（如果 wasmMaxEf>0，会先 clamp）
	 */
	private async ensureResultsCapAtLeast(n: number) {
		if (n <= 0) return;

		if (this.wasmMaxEf > 0) n = Math.min(n, this.wasmMaxEf);

		const cur = await this.wasm.getResultsCap();
		if (cur >= n) return;

		let next = cur > 0 ? cur : 1000;
		while (next < n) next *= 2;

		await this.wasm.setResultsCap(next);
	}
}
