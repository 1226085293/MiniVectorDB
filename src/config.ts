// src/config.ts
import path from "path";
import type {
	DBOpenOptions,
	InternalResolvedConfig,
	ModePreset,
} from "./types";
import type { ModelArchitecture } from "./embedder";

function inferDimFromModel(modelName: string, arch: ModelArchitecture): number {
	return arch === "clip" ? 512 : 384;
}

function resolveArch(
	modelName?: string,
	arch?: ModelArchitecture,
): ModelArchitecture {
	if (arch) return arch;
	if (!modelName) return "text";
	return modelName.toLowerCase().includes("clip") ? "clip" : "text";
}

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

function numEnv(name: string, fallback: number): number {
	const v = Number(process.env[name] ?? "");
	return Number.isFinite(v) && v !== 0 ? v : fallback;
}

export function resolveOpenConfig(opts: DBOpenOptions): InternalResolvedConfig {
	const storageDir =
		opts.storageDir ||
		process.env.MINIVECTOR_STORAGE_DIR ||
		path.join(process.cwd(), "data");

	const collection = opts.collection || process.env.MINIVECTOR_COLLECTION || "";
	const prefix = collection ? `${collection}.` : "";

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

	const deletedRebuildThreshold =
		typeof opts.deletedRebuildThreshold === "number"
			? opts.deletedRebuildThreshold
			: Number(process.env.MINIVECTOR_DELETED_REBUILD_THRESHOLD ?? "0.2");

	const autoRebuildOnLoad =
		typeof opts.autoRebuildOnLoad === "boolean"
			? opts.autoRebuildOnLoad
			: (process.env.MINIVECTOR_AUTO_REBUILD_ON_LOAD ?? "1") !== "0";

	const embeddingCacheSize =
		typeof opts.embeddingCacheSize === "number"
			? opts.embeddingCacheSize
			: Number(process.env.EMBEDDING_CACHE_SIZE ?? "5000");

	return {
		storageDir,
		collection,
		prefix,

		modelName,
		modelArchitecture,

		mode,
		preloadVectors,

		dim,
		capacity,
		seed: opts.seed,

		m: opts.m || Number(process.env.HNSW_M ?? 0) || preset.m,
		ef_construction:
			opts.ef_construction ||
			Number(process.env.HNSW_EF ?? 0) ||
			preset.ef_construction,

		baseEfSearch: numEnv("BASE_EF_SEARCH", preset.baseEfSearch),
		rerankMultiplier: numEnv("RERANK_MULTIPLIER", preset.rerankMultiplier),
		maxAnnK: numEnv("MAX_ANN_K", preset.maxAnnK),
		resultsCap: numEnv("HNSW_RESULTS_CAP", preset.resultsCap),

		deletedRebuildThreshold: Number.isFinite(deletedRebuildThreshold)
			? deletedRebuildThreshold
			: 0.2,
		autoRebuildOnLoad,

		modelCacheDir: opts.modelCacheDir || process.env.MINIVECTOR_MODEL_CACHE_DIR,
		localFilesOnly:
			opts.localFilesOnly ?? process.env.MINIVECTOR_LOCAL_FILES_ONLY === "1",
		embeddingCacheSize: Number.isFinite(embeddingCacheSize)
			? embeddingCacheSize
			: 0,
	};
}
