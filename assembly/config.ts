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
 * @zh-CN 内存偏移量常量
 * @en Memory offset constants.
 * @remarks 我们使用简单的指针递增分配器，起始位置由静态数据之后决定。
 * @remarks We use a simple bump pointer allocator, with allocation starting after static data.
 */
export const NULL_PTR: i32 = 0;

/**
 * @zh-CN 更新核心配置参数。
 *         此函数必须在调用 init_index 或 insert 之前调用。
 * @en Updates the core configuration parameters.
 *       This function must be called before calling init_index or insert.
 * @param dim - 向量的维度 (Vector dimension).
 * @param m - HNSW图中每个节点的最大连接数 (Max number of connections per node in HNSW graph).
 * @param ef_construction - HNSW索引构建时动态候选列表的大小 (Size of the dynamic candidate list during HNSW index construction).
 */
export function update_config(dim: i32, m: i32, ef_construction: i32): void {
	DIM = dim;
	M = m;
	EF_CONSTRUCTION = ef_construction;
	M_MAX0 = m * 2;
}
