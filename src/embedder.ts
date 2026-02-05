import {
	pipeline,
	RawImage,
	AutoProcessor,
	AutoTokenizer,
	CLIPTextModelWithProjection,
	CLIPVisionModelWithProjection,
} from "@xenova/transformers";
import fs from "fs";

export type ModelArchitecture = "text" | "clip";

/**
 * @zh-CN 本地嵌入模型包装类。
 * @en Local embedding model wrapper.
 */
export class LocalEmbedder {
	private pipe: any = null;
	private modelName: string;
	private architecture: ModelArchitecture;

	private processor: any = null;
	private tokenizer: any = null;
	private visionModel: any = null;
	private textModel: any = null;

	private initPromise: Promise<void> | null = null;

	constructor(modelName: string, architecture?: ModelArchitecture) {
		this.modelName = modelName;
		// 如果未指定架构，则根据名称猜测 (向后兼容)
		this.architecture =
			architecture ||
			(modelName.toLowerCase().includes("clip") ? "clip" : "text");
	}

	async init(): Promise<void> {
		if (this.initPromise) return this.initPromise;

		this.initPromise = (async () => {
			try {
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

		const isImageSrc =
			input instanceof Buffer ||
			input instanceof Uint8Array ||
			(typeof input === "string" &&
				(input.startsWith("http") || fs.existsSync(input)));

		// --- CLIP 多模态处理 ---
		if (this.architecture === "clip") {
			if (isImageSrc) {
				const image = await RawImage.read(input as any);
				const { pixel_values } = await this.processor(image);
				const { image_embeds } = await this.visionModel({ pixel_values });
				return this._normalize(image_embeds.data as Float32Array);
			} else {
				const inputs = await this.tokenizer(input, {
					padding: true,
					truncation: true,
				});
				const { text_embeds } = await this.textModel(inputs);
				return this._normalize(text_embeds.data as Float32Array);
			}
		}

		// --- 标准文本模型处理 ---
		const output = await this.pipe(input, { pooling: "mean", normalize: true });
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
