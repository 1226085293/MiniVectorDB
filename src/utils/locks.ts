// src/utils/locks.ts
import type { InternalResolvedConfig } from "../types";

export function lockKeyOf(cfg: InternalResolvedConfig) {
	// 同一个“物理库”的定义：storageDir + prefix
	return `${cfg.storageDir}::${cfg.prefix}`;
}
