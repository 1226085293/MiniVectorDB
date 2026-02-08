// src/embedder.ts
import {
	env,
	pipeline,
	RawImage,
	AutoProcessor,
	AutoTokenizer,
	CLIPTextModelWithProjection,
	CLIPVisionModelWithProjection,
} from "@xenova/transformers";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

export type ModelArchitecture = "text" | "clip";

export type LocalEmbedderOptions = {
	/**
	 * ✅ 模型缓存目录（离线/预热/复用）
	 * - Node 环境下，@xenova/transformers 会把模型权重与 tokenizer 等缓存到 cacheDir。
	 * - 你可以预先把该目录打包到镜像里，实现“离线启动”。
	 */
	cacheDir?: string;

	/**
	 * ✅ 如果你把模型文件预放在 cacheDir，并希望避免联网，可开启：
	 * （不同 transformers 版本行为可能略不同，但 env.localModelPath/env.allowLocalModels 通常可用）
	 */
	localFilesOnly?: boolean;
};

export class LocalEmbedder {
	private pipe: any = null;
	private modelName: string;
	private architecture: ModelArchitecture;

	private processor: any = null;
	private tokenizer: any = null;
	private visionModel: any = null;
	private textModel: any = null;

	private initPromise: Promise<void> | null = null;
	private opts: LocalEmbedderOptions;

	constructor(
		modelName: string,
		architecture?: ModelArchitecture,
		opts: LocalEmbedderOptions = {},
	) {
		this.modelName = modelName;
		this.architecture =
			architecture ||
			(modelName.toLowerCase().includes("clip") ? "clip" : "text");
		this.opts = opts;
	}

	private isLikelyImagePath(p: string): boolean {
		const ext = path.extname(p).toLowerCase();
		return [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"].includes(ext);
	}

	async init(): Promise<void> {
		if (this.initPromise) return this.initPromise;

		this.initPromise = (async () => {
			try {
				if (this.opts.cacheDir) {
					env.cacheDir = this.opts.cacheDir;
				}
				if (this.opts.localFilesOnly) {
					// Best-effort flags (may vary by transformers.js version)
					// @ts-ignore
					env.allowLocalModels = true;
					// @ts-ignore
					env.useBrowserCache = false;
				}

				console.log(
					`Loading embedding model [${this.architecture}]: ${this.modelName}...`,
				);

				if (this.architecture === "clip") {
					const [p, t, v, tm] = await Promise.all([
						AutoProcessor.from_pretrained(this.modelName),
						AutoTokenizer.from_pretrained(this.modelName),
						CLIPVisionModelWithProjection.from_pretrained(this.modelName),
						CLIPTextModelWithProjection.from_pretrained(this.modelName),
					]);
					this.processor = p;
					this.tokenizer = t;
					this.visionModel = v;
					this.textModel = tm;
				} else {
					// @ts-ignore
					this.pipe = await pipeline("feature-extraction", this.modelName);
				}

				console.log("Model loaded successfully.");
			} catch (error) {
				this.initPromise = null;
				throw error;
			}
		})();

		return this.initPromise;
	}

	async embed(input: any): Promise<Float32Array> {
		await this.init();

		let isImageSrc = input instanceof Buffer || input instanceof Uint8Array;
		let normalizedInput = input;

		if (!isImageSrc && typeof input === "string") {
			const s = input;

			if (s.startsWith("http://") || s.startsWith("https://")) {
				isImageSrc = true;
			} else if (s.startsWith("file://")) {
				try {
					const localPath = fileURLToPath(s);
					if (this.isLikelyImagePath(localPath) && fs.existsSync(localPath)) {
						isImageSrc = true;
						normalizedInput = localPath;
					}
				} catch {}
			} else if (fs.existsSync(s) && this.isLikelyImagePath(s)) {
				isImageSrc = true;
			}
		}

		if (this.architecture === "clip") {
			if (isImageSrc) {
				const image = await RawImage.read(normalizedInput as any);
				const { pixel_values } = await this.processor(image);
				const { image_embeds } = await this.visionModel({ pixel_values });
				return this._normalize(image_embeds.data as Float32Array);
			} else {
				const inputs = await this.tokenizer(normalizedInput, {
					padding: true,
					truncation: true,
				});
				const { text_embeds } = await this.textModel(inputs);
				return this._normalize(text_embeds.data as Float32Array);
			}
		}

		const output = await this.pipe(normalizedInput, {
			pooling: "mean",
			normalize: true,
		});
		return output.data as Float32Array;
	}

	private _normalize(v: Float32Array): Float32Array {
		let normSq = 0;
		for (let i = 0; i < v.length; i++) normSq += v[i] * v[i];
		const norm = Math.sqrt(normSq);
		if (norm > 0) {
			for (let i = 0; i < v.length; i++) v[i] /= norm;
		}
		return v;
	}
}
