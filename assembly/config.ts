/**
 * @zh-CN 配置常量
 * @en Configuration constants
 */

/**
 * @zh-CN 向量维度
 * @en Vector dimension.
 * @remarks
 * 根据使用的模型修改此值 (Modify this based on your model):
 * - Xenova/all-MiniLM-L6-v2 (默认文本模型 Default Text): 384
 * - Xenova/clip-vit-base-patch32 (图文多模态模型 Multi-modal CLIP): 512
 *
 * 必须是 4 的倍数以支持 SIMD 优化 (Must be a multiple of 4 for SIMD optimization).
 */
export let DIM: i32 = 384;

/**
 * @zh-CN HNSW 超参数配置
 * @en HNSW Hyperparameters configuration.
 */

/**
 * @zh-CN 图的最大层数
 * @en Maximum number of layers in the graph.
 */
export let MAX_LAYERS: i32 = 4;

/**
 * @zh-CN 每个节点在每层的最大连接数 (M)
 * @en Maximum number of connections per element per layer (M).
 */
export let M: i32 = 16;

/**
 * @zh-CN 第 0 层 (最底层) 的最大连接数
 * @en Maximum connections for layer 0 (usually 2*M).
 */
export let M_MAX0: i32 = 32;

/**
 * @zh-CN 动态候选列表的大小 (efConstruction)
 * @en Size of the dynamic candidate list for construction (efConstruction).
 * @remarks 影响构建索引时的精度和速度，值越大精度越高但构建越慢。
 * @remarks Affects the quality and speed of index construction; higher values lead to better quality but slower construction.
 */
export let EF_CONSTRUCTION: i32 = 100;

/**
 * @zh-CN 搜索阶段的 ef (efSearch)
 * @en ef during search (efSearch).
 * @remarks 值越大召回率越高但查询越慢。通常 >= k。
 */
export let EF_SEARCH: i32 = 50;

/**
 * @zh-CN 内存偏移量常量
 * @en Memory offset constants.
 */
export const NULL_PTR: i32 = 0;

/**
 * @zh-CN 更新核心配置参数。
 *         此函数必须在调用 init_index 或 insert 之前调用。
 * @en Updates the core configuration parameters.
 *       This function must be called before calling init_index or insert.
 */
export function update_config(dim: i32, m: i32, ef_construction: i32): void {
	DIM = dim;
	M = m;
	EF_CONSTRUCTION = ef_construction;
	M_MAX0 = m * 2;
}

/**
 * @zh-CN 更新搜索阶段 ef (efSearch)。
 *         可在运行时调整，不影响索引结构。
 * @en Updates efSearch for query-time. Safe to adjust at runtime.
 */
export function update_search_config(ef_search: i32): void {
	EF_SEARCH = ef_search;
}
