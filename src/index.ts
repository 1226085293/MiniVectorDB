import { WasmBridge } from "./core/wasm-bridge";
import { MetaDB } from "./storage/meta-db";
import { LocalEmbedder } from "./embedder";
import path from "path";

/**
 * @zh-CN 搜索结果项的接口定义。
 * @en Interface defining a search result item.
 */
export interface SearchResult {
	id: string;
	score: number;
	metadata: any;
}

/**
 * @zh-CN 插入项的接口定义。
 * @en Interface defining an item to be inserted.
 */
export interface InsertItem {
	id: string;
	/**
	 * @zh-CN 向量数据、文本或图片输入。
	 * @en Vector data, text, or image input.
	 */
	vector: number[] | Float32Array | string | Buffer | Uint8Array;
	metadata?: any;
}

/**
 * @zh-CN 数据库配置的接口定义。
 * @en Interface defining database configuration.
 */
export interface DBConfig {
	/**
	 * @zh-CN 向量维度
	 * @en Vector dimension.
	 */
	dim?: number;
	/**
	 * @zh-CN 嵌入模型名称。默认 'Xenova/all-MiniLM-L6-v2'。
	 */
	modelName?: string;
	/**
	 * @zh-CN 模型架构类型。
	 *         - 'text': 纯文本模型 (如 BERT, RoBERTa)
	 *         - 'clip': 图文多模态模型 (如 CLIP)
	 *         如果不提供，将根据 modelName 自动推断。
	 * @en Model architecture type.
	 *       - 'text': Pure text model (e.g., BERT, RoBERTa)
	 *       - 'clip': Multi-modal model (e.g., CLIP)
	 */
	modelArchitecture?: "text" | "clip";
	/**
	 * @zh-CN 每个节点在每层的最大连接数 (M)。默认 16。
	 */
	m?: number;
	/**
	 * @zh-CN 动态候选列表的大小 (efConstruction)。默认 100。
	 */
	ef_construction?: number;
	/**
	 * @zh-CN 元数据数据库文件的存储路径。
	 */
	metaDbPath?: string;
}

/**
 * @zh-CN MiniVectorDB 类，提供向量数据库的核心功能。
 */
export class MiniVectorDB {
	private wasm: WasmBridge;
	private meta: MetaDB;
	private embedder: LocalEmbedder;
	private config: DBConfig;
	private isReady: boolean = false;

	constructor(config: DBConfig = {}) {
		this.config = config;
		this.wasm = new WasmBridge();
		this.meta = new MetaDB(config.metaDbPath);
		this.embedder = new LocalEmbedder(
			config.modelName || "Xenova/all-MiniLM-L6-v2",
			config.modelArchitecture,
		);
	}

	async init() {
		if (this.isReady) return;

		const wasmConfig = {
			dim: this.config.dim || 384,
			m: this.config.m || 16,
			ef: this.config.ef_construction || 100,
		};

		await this.wasm.init(wasmConfig);
		await this.meta.ready();
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

		let internalId = -1;
		const existing = this.meta.get(id);
		if (existing) {
			internalId = existing.internal_id;
			this.meta.add(id, internalId, metadata || existing.metadata);
		} else {
			internalId = this.meta.items.count();
			this.meta.add(id, internalId, metadata || {});
		}

		this.wasm.insert(internalId, f32Vec);
	}

	async search(
		query: number[] | Float32Array | string | Buffer | Uint8Array,
		k: number = 10,
		filter?: any,
	): Promise<SearchResult[]> {
		if (!this.isReady) await this.init();
		let f32Vec: Float32Array;

		if (
			typeof query === "string" ||
			query instanceof Buffer ||
			query instanceof Uint8Array
		) {
			f32Vec = await this.embedder.embed(query);
		} else if (query instanceof Float32Array) {
			f32Vec = query;
		} else {
			f32Vec = new Float32Array(query);
		}

		const searchK = filter ? k * 10 : k;
		const rawResults = this.wasm.search(f32Vec, searchK);
		const results: SearchResult[] = [];

		for (const res of rawResults) {
			const item = this.meta.getByInternalId(res.id);
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

			results.push({
				id: item.external_id,
				score: res.dist,
				metadata: item.metadata,
			});

			if (results.length >= k) break;
		}
		return results;
	}

	async save(filepath: string) {
		await this.wasm.save(filepath);
	}

	async load(filepath: string) {
		if (!this.isReady) await this.init();
		await this.wasm.load(filepath);
	}

	getStats() {
		return {
			memory: this.wasm["memory"].buffer.byteLength,
			items: this.meta.items.count(),
		};
	}

	async close() {
		await this.meta.close();
	}
}
