// assembly/hnsw.ts
import { dist_l2_sq } from "./math";
import { alloc } from "./memory";
import {
	M,
	M_MAX0,
	EF_CONSTRUCTION,
	EF_SEARCH,
	MAX_LAYERS,
	DIM,
} from "./config";
import { get_node_size, get_vector_size } from "./types";

// --- GLOBALS ---
let entry_point_id: i32 = -1;
let max_level: i32 = -1;
let element_count: i32 = 0;

let node_offsets_ptr: usize = 0;
let vector_storage_ptr: usize = 0;
let results_buffer_ptr: usize = 0;
let max_elements: i32 = 0;

// --- VISITED (STAMP) ---
let visited_mark_ptr: usize = 0; // i32[max_elements]
let visited_stamp: i32 = 1;

// --- HEAP BUFFERS (REUSED, NO GC) ---
const MAX_EF: i32 = 4096;

// candidates (min-heap)
let cand_ids_ptr: usize = 0; // i32[MAX_EF]
let cand_dist_ptr: usize = 0; // f32[MAX_EF]

// results (max-heap, worst at root)
let res_ids_ptr: usize = 0; // i32[MAX_EF]
let res_dist_ptr: usize = 0; // f32[MAX_EF]

// output buffers (closest first)
let out_ids_ptr: usize = 0; // i32[MAX_EF]
let out_dist_ptr: usize = 0; // f32[MAX_EF]

// marks for partial extraction (no alloc in hot path)
let used_mark_ptr: usize = 0; // u8[MAX_EF]

// neighbor selection scratch
let sel_ids_ptr: usize = 0; // i32[MAX_EF]
let sel_count: i32 = 0;

// --- HELPERS ---
function get_node_ptr(id: i32): usize {
	return load<usize>(node_offsets_ptr + <usize>id * 4);
}
function get_vector_ptr(id: i32): usize {
	return vector_storage_ptr + <usize>id * get_vector_size();
}
function visited_get(id: i32): bool {
	return load<i32>(visited_mark_ptr + <usize>id * 4) == visited_stamp;
}
function visited_set(id: i32): void {
	store<i32>(visited_mark_ptr + <usize>id * 4, visited_stamp);
}
function visited_next_stamp(): void {
	visited_stamp++;
	if (visited_stamp == 0) {
		visited_stamp = 1;
		for (let i: i32 = 0; i < max_elements; i++) {
			store<i32>(visited_mark_ptr + <usize>i * 4, 0);
		}
	}
}

// ✅ Expose whether a node exists in the graph (used by host to decide insert vs update)
export function has_node(id: i32): bool {
	if (id < 0 || id >= max_elements) return false;
	return load<usize>(node_offsets_ptr + <usize>id * 4) != 0;
}

// --- MIN-HEAP (candidates) ---
function cand_swap(i: i32, j: i32): void {
	let id_i = load<i32>(cand_ids_ptr + <usize>i * 4);
	let id_j = load<i32>(cand_ids_ptr + <usize>j * 4);
	let d_i = load<f32>(cand_dist_ptr + <usize>i * 4);
	let d_j = load<f32>(cand_dist_ptr + <usize>j * 4);

	store<i32>(cand_ids_ptr + <usize>i * 4, id_j);
	store<i32>(cand_ids_ptr + <usize>j * 4, id_i);
	store<f32>(cand_dist_ptr + <usize>i * 4, d_j);
	store<f32>(cand_dist_ptr + <usize>j * 4, d_i);
}

function cand_push(size: i32, id: i32, dist: f32): i32 {
	let i = size;
	store<i32>(cand_ids_ptr + <usize>i * 4, id);
	store<f32>(cand_dist_ptr + <usize>i * 4, dist);
	size++;

	while (i > 0) {
		let p = (i - 1) >> 1;
		let d_p = load<f32>(cand_dist_ptr + <usize>p * 4);
		let d_i = load<f32>(cand_dist_ptr + <usize>i * 4);
		if (d_p <= d_i) break;
		cand_swap(i, p);
		i = p;
	}
	return size;
}

let tmp_pop_id: i32 = -1;
let tmp_pop_dist: f32 = 0.0;

function cand_pop(size: i32): i32 {
	if (size <= 0) {
		tmp_pop_id = -1;
		tmp_pop_dist = 0.0;
		return 0;
	}

	tmp_pop_id = load<i32>(cand_ids_ptr + 0);
	tmp_pop_dist = load<f32>(cand_dist_ptr + 0);

	size--;
	if (size == 0) return 0;

	let last_id = load<i32>(cand_ids_ptr + <usize>size * 4);
	let last_d = load<f32>(cand_dist_ptr + <usize>size * 4);
	store<i32>(cand_ids_ptr + 0, last_id);
	store<f32>(cand_dist_ptr + 0, last_d);

	let i: i32 = 0;
	while (true) {
		let l = (i << 1) + 1;
		if (l >= size) break;
		let r = l + 1;

		let smallest = l;
		let d_l = load<f32>(cand_dist_ptr + <usize>l * 4);
		if (r < size) {
			let d_r = load<f32>(cand_dist_ptr + <usize>r * 4);
			if (d_r < d_l) smallest = r;
		}

		let d_small = load<f32>(cand_dist_ptr + <usize>smallest * 4);
		let d_i = load<f32>(cand_dist_ptr + <usize>i * 4);
		if (d_i <= d_small) break;

		cand_swap(i, smallest);
		i = smallest;
	}

	return size;
}

// --- MAX-HEAP (results, worst at root) ---
function res_peek_worst(size: i32): f32 {
	if (size <= 0) return 99999999.0;
	return load<f32>(res_dist_ptr + 0);
}

function res_swap(i: i32, j: i32): void {
	let id_i = load<i32>(res_ids_ptr + <usize>i * 4);
	let id_j = load<i32>(res_ids_ptr + <usize>j * 4);
	let d_i = load<f32>(res_dist_ptr + <usize>i * 4);
	let d_j = load<f32>(res_dist_ptr + <usize>j * 4);

	store<i32>(res_ids_ptr + <usize>i * 4, id_j);
	store<i32>(res_ids_ptr + <usize>j * 4, id_i);
	store<f32>(res_dist_ptr + <usize>i * 4, d_j);
	store<f32>(res_dist_ptr + <usize>j * 4, d_i);
}

function res_push(size: i32, id: i32, dist: f32): i32 {
	let i = size;
	store<i32>(res_ids_ptr + <usize>i * 4, id);
	store<f32>(res_dist_ptr + <usize>i * 4, dist);
	size++;

	while (i > 0) {
		let p = (i - 1) >> 1;
		let d_p = load<f32>(res_dist_ptr + <usize>p * 4);
		let d_i = load<f32>(res_dist_ptr + <usize>i * 4);
		if (d_p >= d_i) break;
		res_swap(i, p);
		i = p;
	}
	return size;
}

function res_pop(size: i32): i32 {
	if (size <= 0) {
		tmp_pop_id = -1;
		tmp_pop_dist = 0.0;
		return 0;
	}

	tmp_pop_id = load<i32>(res_ids_ptr + 0);
	tmp_pop_dist = load<f32>(res_dist_ptr + 0);

	size--;
	if (size == 0) return 0;

	let last_id = load<i32>(res_ids_ptr + <usize>size * 4);
	let last_d = load<f32>(res_dist_ptr + <usize>size * 4);
	store<i32>(res_ids_ptr + 0, last_id);
	store<f32>(res_dist_ptr + 0, last_d);

	let i: i32 = 0;
	while (true) {
		let l = (i << 1) + 1;
		if (l >= size) break;
		let r = l + 1;

		let largest = l;
		let d_l = load<f32>(res_dist_ptr + <usize>l * 4);
		if (r < size) {
			let d_r = load<f32>(res_dist_ptr + <usize>r * 4);
			if (d_r > d_l) largest = r;
		}

		let d_large = load<f32>(res_dist_ptr + <usize>largest * 4);
		let d_i = load<f32>(res_dist_ptr + <usize>i * 4);
		if (d_i >= d_large) break;

		res_swap(i, largest);
		i = largest;
	}
	return size;
}

// replace root (worst) with better element and sift-down (faster than pop+push)
function res_replace_root(size: i32, id: i32, dist: f32): void {
	store<i32>(res_ids_ptr + 0, id);
	store<f32>(res_dist_ptr + 0, dist);

	let i: i32 = 0;
	while (true) {
		let l = (i << 1) + 1;
		if (l >= size) break;
		let r = l + 1;

		let largest = l;
		let d_l = load<f32>(res_dist_ptr + <usize>l * 4);
		if (r < size) {
			let d_r = load<f32>(res_dist_ptr + <usize>r * 4);
			if (d_r > d_l) largest = r;
		}

		let d_large = load<f32>(res_dist_ptr + <usize>largest * 4);
		let d_i = load<f32>(res_dist_ptr + <usize>i * 4);
		if (d_i >= d_large) break;

		res_swap(i, largest);
		i = largest;
	}
}

// full heap-sort (used only for SEARCH results)
function res_to_sorted_full(size: i32): i32 {
	let n = size;
	for (let i = n - 1; i >= 0; i--) {
		size = res_pop(size);
		store<i32>(out_ids_ptr + <usize>i * 4, tmp_pop_id);
		store<f32>(out_dist_ptr + <usize>i * 4, tmp_pop_dist);
	}
	return n;
}

// partial extraction: get k smallest from res arrays (O(k*ef)), k is tiny in build (<=64)
// writes ascending order into out buffers.
function res_extract_k_smallest_sorted(res_size: i32, k: i32): i32 {
	if (k <= 0 || res_size <= 0) return 0;
	if (k > res_size) k = res_size;
	if (k > MAX_EF) k = MAX_EF;

	// mark clear for [0..res_size)
	for (let i: i32 = 0; i < res_size; i++) {
		store<u8>(used_mark_ptr + <usize>i, 0);
	}

	for (let out: i32 = 0; out < k; out++) {
		let best_idx: i32 = -1;
		let best_dist: f32 = 99999999.0;

		for (let i: i32 = 0; i < res_size; i++) {
			if (load<u8>(used_mark_ptr + <usize>i) != 0) continue;
			let d = load<f32>(res_dist_ptr + <usize>i * 4);
			if (d < best_dist) {
				best_dist = d;
				best_idx = i;
			}
		}

		if (best_idx < 0) break;
		store<u8>(used_mark_ptr + <usize>best_idx, 1);

		let id = load<i32>(res_ids_ptr + <usize>best_idx * 4);
		store<i32>(out_ids_ptr + <usize>out * 4, id);
		store<f32>(out_dist_ptr + <usize>out * 4, best_dist);
	}

	return k;
}

// --- INIT ---
export function init_index(capacity: i32): void {
	max_elements = capacity;

	node_offsets_ptr = alloc(<usize>capacity * 4);
	vector_storage_ptr = alloc(<usize>capacity * get_vector_size());
	results_buffer_ptr = alloc(1000 * 8);

	visited_mark_ptr = alloc(<usize>capacity * 4);

	cand_ids_ptr = alloc(<usize>MAX_EF * 4);
	cand_dist_ptr = alloc(<usize>MAX_EF * 4);
	res_ids_ptr = alloc(<usize>MAX_EF * 4);
	res_dist_ptr = alloc(<usize>MAX_EF * 4);

	out_ids_ptr = alloc(<usize>MAX_EF * 4);
	out_dist_ptr = alloc(<usize>MAX_EF * 4);

	used_mark_ptr = alloc(<usize>MAX_EF); // u8

	sel_ids_ptr = alloc(<usize>MAX_EF * 4);

	entry_point_id = -1;
	max_level = -1;
	element_count = 0;
	visited_stamp = 1;

	for (let i: i32 = 0; i < capacity; i++) {
		store<usize>(node_offsets_ptr + <usize>i * 4, 0);
		store<i32>(visited_mark_ptr + <usize>i * 4, 0);
	}
	for (let i: i32 = 0; i < MAX_EF; i++) {
		store<u8>(used_mark_ptr + <usize>i, 0);
	}
}

function get_random_level(): i32 {
	let level: i32 = 0;
	while (Math.random() < 0.5 && level < MAX_LAYERS - 1) {
		level++;
	}
	return level;
}

// --- SEARCH LAYER (stable-quality core) ---
// Always maintains full ef "best" set in res heap (quality is NOT reduced).
// Export path:
//  - For query search: full sort output
//  - For build/insert: partial extract only (no full sort), but still based on full ef best set.
function search_layer(
	query_vec_ptr: usize,
	entry_node: i32,
	layer: i32,
	ef: i32,
	buildMode: bool,
	target: i32,
): i32 {
	if (ef <= 0) return 0;
	if (ef > MAX_EF) ef = MAX_EF;

	visited_next_stamp();

	// frontier capacity slightly larger than ef improves convergence and stability
	let cand_cap: i32 = ef * 2 + 32;
	if (cand_cap > MAX_EF) cand_cap = MAX_EF;
	if (cand_cap < ef) cand_cap = ef;

	let cand_size: i32 = 0;
	let res_size: i32 = 0;

	let dist0 = dist_l2_sq(query_vec_ptr, get_vector_ptr(entry_node));

	visited_set(entry_node);
	cand_size = cand_push(cand_size, entry_node, dist0);
	res_size = res_push(res_size, entry_node, dist0);

	while (cand_size > 0) {
		cand_size = cand_pop(cand_size);
		let curr_id = tmp_pop_id;
		let curr_dist = tmp_pop_dist;

		if (res_size >= ef && curr_dist > res_peek_worst(res_size)) break;

		let curr_node_ptr = get_node_ptr(curr_id);
		if (curr_node_ptr == 0) continue;

		let runner_ptr = curr_node_ptr + 8;
		for (let l = 0; l < layer; l++) {
			let cap = l == 0 ? M_MAX0 : M;
			runner_ptr += 4 + <usize>cap * 4;
		}

		let neighbor_count = load<i32>(runner_ptr);
		let neighbors_start = runner_ptr + 4;

		for (let i = 0; i < neighbor_count; i++) {
			let neighbor_id = load<i32>(neighbors_start + <usize>i * 4);
			if (neighbor_id < 0 || neighbor_id >= max_elements) continue;
			if (visited_get(neighbor_id)) continue;

			visited_set(neighbor_id);

			let d = dist_l2_sq(query_vec_ptr, get_vector_ptr(neighbor_id));

			if (res_size < ef || d < res_peek_worst(res_size)) {
				if (cand_size < cand_cap) {
					cand_size = cand_push(cand_size, neighbor_id, d);
				}

				if (res_size < ef) {
					res_size = res_push(res_size, neighbor_id, d);
				} else {
					if (d < res_peek_worst(res_size)) {
						res_replace_root(res_size, neighbor_id, d);
					}
				}
			}
		}
	}

	if (!buildMode) {
		return res_to_sorted_full(res_size);
	}

	// build: we only need a candidate pool for neighbor heuristic selection.
	// IMPORTANT: pool extraction is from the FULL ef-best set in res heap (quality preserved).
	let pool: i32 = target * 2;
	if (pool < target) pool = target;
	if (pool > ef) pool = ef;
	if (pool > res_size) pool = res_size;

	return res_extract_k_smallest_sorted(res_size, pool);
}

// --- Heuristic neighbor selection (diversity-aware) ---
function select_neighbors_heuristic(
	query_vec_ptr: usize,
	found: i32,
	target: i32,
): i32 {
	sel_count = 0;
	if (found <= 0) return 0;
	if (target <= 0) return 0;
	if (target > MAX_EF) target = MAX_EF;

	for (let i: i32 = 0; i < found; i++) {
		let cand_id = load<i32>(out_ids_ptr + <usize>i * 4);
		let d_cq = load<f32>(out_dist_ptr + <usize>i * 4);

		let accept = true;
		let cand_vec = get_vector_ptr(cand_id);

		for (let j: i32 = 0; j < sel_count; j++) {
			let s_id = load<i32>(sel_ids_ptr + <usize>j * 4);
			let d_cs = dist_l2_sq(cand_vec, get_vector_ptr(s_id));
			if (d_cs < d_cq) {
				accept = false;
				break;
			}
		}

		if (accept) {
			store<i32>(sel_ids_ptr + <usize>sel_count * 4, cand_id);
			sel_count++;
			if (sel_count >= target) break;
		}
	}

	// fill remaining from best list
	if (sel_count < target) {
		for (let i: i32 = 0; i < found && sel_count < target; i++) {
			let cand_id = load<i32>(out_ids_ptr + <usize>i * 4);

			let exists = false;
			for (let j: i32 = 0; j < sel_count; j++) {
				if (load<i32>(sel_ids_ptr + <usize>j * 4) == cand_id) {
					exists = true;
					break;
				}
			}
			if (exists) continue;

			store<i32>(sel_ids_ptr + <usize>sel_count * 4, cand_id);
			sel_count++;
		}
	}

	return sel_count;
}

// --- CONNECTION LOGIC ---
function add_connection(src: i32, dst: i32, layer: i32): void {
	let ptr = get_node_ptr(src);
	let runner = ptr + 8;
	for (let l = 0; l < layer; l++) {
		let cap = l == 0 ? M_MAX0 : M;
		runner += 4 + <usize>cap * 4;
	}

	let count = load<i32>(runner);
	let cap_cur = layer == 0 ? M_MAX0 : M;
	let neighbors_ptr = runner + 4;

	for (let i = 0; i < count; i++) {
		if (load<i32>(neighbors_ptr + <usize>i * 4) == dst) return;
	}

	if (count < cap_cur) {
		store<i32>(neighbors_ptr + <usize>count * 4, dst);
		store<i32>(runner, count + 1);
	} else {
		let src_vec = get_vector_ptr(src);
		let worst_idx = -1;
		let worst_dist: f32 = -1.0;

		for (let i = 0; i < count; i++) {
			let n_id = load<i32>(neighbors_ptr + <usize>i * 4);
			let d = dist_l2_sq(src_vec, get_vector_ptr(n_id));
			if (d > worst_dist) {
				worst_dist = d;
				worst_idx = i;
			}
		}

		let dst_dist = dist_l2_sq(src_vec, get_vector_ptr(dst));
		if (dst_dist < worst_dist && worst_idx >= 0) {
			store<i32>(neighbors_ptr + <usize>worst_idx * 4, dst);
		}
	}
}

// --- INSERT (new id only) ---
export function insert(id: i32, vector_data_offset: usize): void {
	if (id < 0 || id >= max_elements) return;

	let target_vec_ptr = get_vector_ptr(id);
	memory.copy(target_vec_ptr, vector_data_offset, get_vector_size());

	let level = get_random_level();

	let node_sz = get_node_size(level);
	let node_ptr = alloc(node_sz);
	store<i32>(node_ptr, id);
	store<i32>(node_ptr + 4, level);

	let runner = node_ptr + 8;
	for (let l = 0; l <= level; l++) {
		store<i32>(runner, 0);
		let cap = l == 0 ? M_MAX0 : M;
		runner += 4 + <usize>cap * 4;
	}

	store<usize>(node_offsets_ptr + <usize>id * 4, node_ptr);

	if (entry_point_id == -1) {
		entry_point_id = id;
		max_level = level;
		element_count++;
		return;
	}

	let curr_obj = entry_point_id;
	let curr_dist = dist_l2_sq(target_vec_ptr, get_vector_ptr(curr_obj));

	// Greedy descent
	for (let l = max_level; l > level; l--) {
		let changed = true;
		while (changed) {
			changed = false;

			let c_ptr = get_node_ptr(curr_obj);
			let r_ptr = c_ptr + 8;
			for (let k = 0; k < l; k++) {
				let cap = k == 0 ? M_MAX0 : M;
				r_ptr += 4 + <usize>cap * 4;
			}
			let count = load<i32>(r_ptr);
			let n_ptr = r_ptr + 4;

			for (let i = 0; i < count; i++) {
				let neighbor = load<i32>(n_ptr + <usize>i * 4);
				let d = dist_l2_sq(target_vec_ptr, get_vector_ptr(neighbor));
				if (d < curr_dist) {
					curr_dist = d;
					curr_obj = neighbor;
					changed = true;
				}
			}
		}
	}

	// Connect down
	for (let l = level < max_level ? level : max_level; l >= 0; l--) {
		let target = l == 0 ? M_MAX0 : M;

		let found = search_layer(
			target_vec_ptr,
			curr_obj,
			l,
			EF_CONSTRUCTION,
			true,
			target,
		);

		let picked = select_neighbors_heuristic(target_vec_ptr, found, target);

		for (let i = 0; i < picked; i++) {
			let nid = load<i32>(sel_ids_ptr + <usize>i * 4);
			add_connection(id, nid, l);
			add_connection(nid, id, l);
		}

		if (found > 0) {
			curr_obj = load<i32>(out_ids_ptr + 0);
		}
	}

	if (level > max_level) {
		max_level = level;
		entry_point_id = id;
	}
	element_count++;
}

// --- UPDATE VECTOR ONLY (no graph mutation) ---
export function update_vector(id: i32, vector_data_offset: usize): void {
	if (id < 0 || id >= max_elements) return;
	let target_vec_ptr = get_vector_ptr(id);
	memory.copy(target_vec_ptr, vector_data_offset, get_vector_size());
}

// --- SEARCH API ---
export function get_results_ptr(): usize {
	return results_buffer_ptr;
}

export function search(query_vec_offset: usize, k: i32): i32 {
	if (entry_point_id == -1) return 0;
	if (k <= 0) return 0;

	let curr_obj = entry_point_id;
	let curr_dist = dist_l2_sq(query_vec_offset, get_vector_ptr(curr_obj));

	// Greedy descent on upper layers
	for (let l = max_level; l > 0; l--) {
		let changed = true;
		while (changed) {
			changed = false;
			let c_ptr = get_node_ptr(curr_obj);
			let r_ptr = c_ptr + 8;
			for (let kk = 0; kk < l; kk++) {
				let cap = kk == 0 ? M_MAX0 : M;
				r_ptr += 4 + <usize>cap * 4;
			}
			let count = load<i32>(r_ptr);
			let n_ptr = r_ptr + 4;

			for (let i = 0; i < count; i++) {
				let neighbor = load<i32>(n_ptr + <usize>i * 4);
				let d = dist_l2_sq(query_vec_offset, get_vector_ptr(neighbor));
				if (d < curr_dist) {
					curr_dist = d;
					curr_obj = neighbor;
					changed = true;
				}
			}
		}
	}

	let ef = EF_SEARCH;
	if (ef < k) ef = k;
	if (ef > MAX_EF) ef = MAX_EF;

	let found = search_layer(query_vec_offset, curr_obj, 0, ef, false, 0);

	let out_count = found < k ? found : k;
	if (out_count > 1000) out_count = 1000;

	for (let i = 0; i < out_count; i++) {
		let id = load<i32>(out_ids_ptr + <usize>i * 4);
		let d = load<f32>(out_dist_ptr + <usize>i * 4);
		store<i32>(results_buffer_ptr + <usize>i * 8, id);
		store<f32>(results_buffer_ptr + <usize>i * 8 + 4, d);
	}

	return out_count;
}

/* ------------------------------
   ✅ Dump V1 (unchanged here)
--------------------------------*/

const MAGIC_HNSW: i32 = 0x57534e48; // "HNSW"
const DUMP_VERSION: i32 = 1;

export function get_index_dump_size(): usize {
	let size: usize = 40;
	size += <usize>element_count * get_vector_size();

	for (let i: i32 = 0; i < element_count; i++) {
		let node_ptr = get_node_ptr(i);
		let level = load<i32>(node_ptr + 4);

		size += 8;
		for (let l: i32 = 0; l <= level; l++) {
			let cap = l == 0 ? M_MAX0 : M;
			size += 4;
			size += <usize>cap * 4;
		}
	}
	return size;
}

export function save_index(ptr: usize): usize {
	let off: usize = 0;

	store<i32>(ptr + off, MAGIC_HNSW);
	off += 4;
	store<i32>(ptr + off, DUMP_VERSION);
	off += 4;

	store<i32>(ptr + off, DIM);
	off += 4;
	store<i32>(ptr + off, M);
	off += 4;
	store<i32>(ptr + off, EF_CONSTRUCTION);
	off += 4;
	store<i32>(ptr + off, MAX_LAYERS);
	off += 4;

	store<i32>(ptr + off, max_elements);
	off += 4;
	store<i32>(ptr + off, element_count);
	off += 4;
	store<i32>(ptr + off, entry_point_id);
	off += 4;
	store<i32>(ptr + off, max_level);
	off += 4;

	let vec_bytes: usize = <usize>element_count * get_vector_size();
	memory.copy(ptr + off, vector_storage_ptr, vec_bytes);
	off += vec_bytes;

	for (let i: i32 = 0; i < element_count; i++) {
		let node_ptr = get_node_ptr(i);
		let id = load<i32>(node_ptr + 0);
		let level = load<i32>(node_ptr + 4);

		store<i32>(ptr + off, id);
		off += 4;
		store<i32>(ptr + off, level);
		off += 4;

		let runner = node_ptr + 8;
		for (let l: i32 = 0; l <= level; l++) {
			let cap = l == 0 ? M_MAX0 : M;

			let count = load<i32>(runner);
			store<i32>(ptr + off, count);
			off += 4;

			memory.copy(ptr + off, runner + 4, <usize>cap * 4);
			off += <usize>cap * 4;

			runner += 4 + <usize>cap * 4;
		}
	}

	return off;
}

export function load_index(ptr: usize, size: usize): void {
	let off: usize = 0;

	let magic = load<i32>(ptr + off);
	off += 4;
	if (magic != MAGIC_HNSW) {
		entry_point_id = -1;
		max_level = -1;
		element_count = 0;
		return;
	}

	let ver = load<i32>(ptr + off);
	off += 4;
	if (ver != DUMP_VERSION) {
		entry_point_id = -1;
		max_level = -1;
		element_count = 0;
		return;
	}

	let dump_dim = load<i32>(ptr + off);
	off += 4;
	let dump_m = load<i32>(ptr + off);
	off += 4;
	let dump_ef = load<i32>(ptr + off);
	off += 4;
	let dump_layers = load<i32>(ptr + off);
	off += 4;

	let dump_max_elements = load<i32>(ptr + off);
	off += 4;
	let dump_element_count = load<i32>(ptr + off);
	off += 4;
	let dump_entry = load<i32>(ptr + off);
	off += 4;
	let dump_max_level = load<i32>(ptr + off);
	off += 4;

	if (
		dump_dim != DIM ||
		dump_m != M ||
		dump_ef != EF_CONSTRUCTION ||
		dump_layers != MAX_LAYERS
	) {
		entry_point_id = -1;
		max_level = -1;
		element_count = 0;
		return;
	}

	init_index(dump_max_elements);

	element_count = dump_element_count;
	entry_point_id = dump_entry;
	max_level = dump_max_level;

	let vec_bytes: usize = <usize>element_count * get_vector_size();
	memory.copy(vector_storage_ptr, ptr + off, vec_bytes);
	off += vec_bytes;

	for (let i: i32 = 0; i < element_count; i++) {
		let id = load<i32>(ptr + off);
		off += 4;
		let level = load<i32>(ptr + off);
		off += 4;

		let node_sz = get_node_size(level);
		let node_ptr = alloc(node_sz);

		store<i32>(node_ptr + 0, id);
		store<i32>(node_ptr + 4, level);

		let runner = node_ptr + 8;
		for (let l: i32 = 0; l <= level; l++) {
			let cap = l == 0 ? M_MAX0 : M;

			let count = load<i32>(ptr + off);
			off += 4;
			store<i32>(runner, count);

			memory.copy(runner + 4, ptr + off, <usize>cap * 4);
			off += <usize>cap * 4;

			runner += 4 + <usize>cap * 4;
		}

		store<usize>(node_offsets_ptr + <usize>i * 4, node_ptr);
	}
}
