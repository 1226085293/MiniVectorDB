// src/types.ts
import type { ModelArchitecture } from "./embedder";

export type ModePreset = "fast" | "balanced" | "accurate";
export type ScoreMode = "l2" | "cosine" | "similarity";

export interface SearchResult<TMeta = any> {
	id: string;
	score: number;
	metadata: TMeta;
}

export interface InsertItem<TMeta = any> {
	id: string;
	input: number[] | Float32Array | string | Buffer | Uint8Array;
	metadata?: TMeta;
}

export interface SearchOptions<TMeta = any> {
	topK?: number;
	filter?: any | ((metadata: TMeta) => boolean);
	score?: ScoreMode;
}

export interface InsertManyOptions {
	onProgress?: (done: number, total: number) => void;
}

export interface UpdateMetadataOptions {
	merge?: boolean; // default true
}

export interface RebuildOptions {
	capacity?: number;
	persist?: boolean; // default true
	compact?: boolean; // default true (true compact); auto-trigger uses false
	onProgress?: (done: number, total: number) => void;
}

export interface ExportJSONLOptions {
	includeDeleted?: boolean;
	includeVectors?: boolean;
	onProgress?: (done: number, total: number) => void;
}

export interface ImportJSONLOptions {
	batchSize?: number; // default 256
	onProgress?: (done: number, total: number) => void; // total unknown => -1
}

export interface EmbedderLike {
	embed(input: any): Promise<Float32Array> | Float32Array;
	init?: () => Promise<void> | void;
}

export interface DBOpenOptions {
	storageDir?: string;
	collection?: string;

	modelName?: string;
	modelArchitecture?: ModelArchitecture;

	// DI: custom embedder
	embedder?: EmbedderLike;

	// LocalEmbedder only
	modelCacheDir?: string;
	localFilesOnly?: boolean;

	// embedding cache (text only)
	embeddingCacheSize?: number;

	mode?: ModePreset;

	dim?: number;
	capacity?: number;
	preloadVectors?: boolean;
	seed?: number;

	m?: number;
	ef_construction?: number;

	deletedRebuildThreshold?: number;
	autoRebuildOnLoad?: boolean;
}

export type InternalResolvedConfig = {
	storageDir: string;
	collection: string;
	prefix: string;

	modelName: string;
	modelArchitecture: ModelArchitecture;

	mode: ModePreset;
	preloadVectors: boolean;

	dim: number;
	capacity: number;
	seed?: number;

	m: number;
	ef_construction: number;

	baseEfSearch: number;
	rerankMultiplier: number;
	maxAnnK: number;
	resultsCap: number;

	deletedRebuildThreshold: number;
	autoRebuildOnLoad: boolean;

	modelCacheDir?: string;
	localFilesOnly?: boolean;
	embeddingCacheSize: number;
};
