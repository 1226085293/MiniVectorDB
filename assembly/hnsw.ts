// assembly/hnsw.ts
// @ts-nocheck

import { dist_l2_sq } from "./math";
import { alloc } from "./memory";
import { M, M_MAX0, EF_CONSTRUCTION, EF_SEARCH, MAX_LAYERS, DIM, mark_index_inited, mark_index_reset } from "./config";
import { get_node_size, get_vector_size } from "./types";

// --- GLOBALS ---
let entry_point_id: i32 = -1;
let max_level: i32 = -1;
let element_count: i32 = 0; // number of PRESENT nodes (not inserts)
let node_offsets_ptr: usize = 0;
let vector_storage_ptr: usize = 0;

// ✅ results buffer is FIXED to MAX_EF (prevents set_results_cap overflow)
let results_buffer_ptr: usize = 0;
let results_cap: i32 = 1000; // logical cap (clamped <= MAX_EF)

let max_elements: i32 = 0;

// --- VISITED (STAMP) ---
let visited_mark_ptr: usize = 0; // u32[max_elements]
let visited_stamp: u32 = 1;

// --- EF clamp monitor ---
let ef_clamped_flag: i32 = 0;

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

// ✅ FIX: dedicated old-neighbors buffer (prevents overwrite by search_layer)
let old_ids_ptr: usize = 0; // i32[MAX_EF]

// --- RNG (xorshift32) ---
let rng_state: u32 = 2463534242; // default non-zero

export function seed_rng(seed: u32): void {
	rng_state = seed != 0 ? seed : 2463534242;
}

@inline
function xorshift32(): u32 {
	let x = rng_state;
	x ^= x << 13;
	x ^= x >>> 17;
	x ^= x << 5;
	rng_state = x;
	return x;
}

@inline
function rand01(): f64 {
	let r = xorshift32() >>> 8;
	return <f64>r / <f64>0x01000000;
}

// --- CONFIG API ---
export function set_results_cap(cap: i32): void {
	// ✅ safe: cap only controls how many items host will read,
	// result buffer is fixed to MAX_EF, and search clamps out_count.
	if (cap <= 0) return;
	if (cap > MAX_EF) cap = MAX_EF;
	results_cap = cap;
}

export function get_results_cap(): i32 {
	return results_cap;
}

export function get_max_ef(): i32 {
	return MAX_EF;
}

export function was_ef_clamped(): i32 {
	return ef_clamped_flag;
}

export function clear_ef_clamped(): void {
	ef_clamped_flag = 0;
}

// ✅ expose capacity to host (hard safety checks)
export function get_max_elements(): i32 {
	return max_elements;
}

// --- HELPERS ---
function get_node_ptr(id: i32): usize {
	return load<usize>(node_offsets_ptr + <usize>id * 4);
}
function set_node_ptr(id: i32, ptr: usize): void {
	store<usize>(node_offsets_ptr + <usize>id * 4, ptr);
}
function get_vector_ptr(id: i32): usize {
	return vector_storage_ptr + <usize>id * get_vector_size();
}

function visited_get(id: i32): bool {
	return load<u32>(visited_mark_ptr + <usize>id * 4) == visited_stamp;
}
function visited_set(id: i32): void {
	store<u32>(visited_mark_ptr + <usize>id * 4, visited_stamp);
}
function visited_next_stamp(): void {
	visited_stamp++;
	if (visited_stamp == 0) {
		visited_stamp = 1;
		for (let i: i32 = 0; i < max_elements; i++) {
			store<u32>(visited_mark_ptr + <usize>i * 4, 0);
		}
	}
}

// ✅ Expose whether a node exists in the graph (used by host to decide insert vs update)
export function has_node(id: i32): bool {
	if (id < 0 || id >= max_elements) return false;
	return get_node_ptr(id) != 0;
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

function res_to_sorted_full(size: i32): i32 {
	let n = size;
	for (let i = n - 1; i >= 0; i--) {
		size = res_pop(size);
		store<i32>(out_ids_ptr + <usize>i * 4, tmp_pop_id);
		store<f32>(out_dist_ptr + <usize>i * 4, tmp_pop_dist);
	}
	return n;
}

function res_extract_k_smallest_sorted(res_size: i32, k: i32): i32 {
	if (k <= 0 || res_size <= 0) return 0;
	if (k > res_size) k = res_size;
	if (k > MAX_EF) k = MAX_EF;

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
	if (capacity <= 0) {
		entry_point_id = -1;
		max_level = -1;
		element_count = 0;
		max_elements = 0;
		mark_index_reset();
		return;
	}

	max_elements = capacity;

	// alloc now aligns to 16 by default (memory.ts)
	node_offsets_ptr = alloc(<usize>capacity * 4);
	vector_storage_ptr = alloc(<usize>capacity * get_vector_size());

	// ✅ results buffer fixed to MAX_EF always
	if (results_cap <= 0) results_cap = 1000;
	if (results_cap > MAX_EF) results_cap = MAX_EF;
	results_buffer_ptr = alloc(<usize>MAX_EF * 8);

	visited_mark_ptr = alloc(<usize>capacity * 4);

	cand_ids_ptr = alloc(<usize>MAX_EF * 4);
	cand_dist_ptr = alloc(<usize>MAX_EF * 4);
	res_ids_ptr = alloc(<usize>MAX_EF * 4);
	res_dist_ptr = alloc(<usize>MAX_EF * 4);

	out_ids_ptr = alloc(<usize>MAX_EF * 4);
	out_dist_ptr = alloc(<usize>MAX_EF * 4);

	used_mark_ptr = alloc(<usize>MAX_EF); // u8
	sel_ids_ptr = alloc(<usize>MAX_EF * 4);

	// ✅ FIX: old-neighbors buffer
	old_ids_ptr = alloc(<usize>MAX_EF * 4);

	entry_point_id = -1;
	max_level = -1;
	element_count = 0;
	visited_stamp = 1;
	ef_clamped_flag = 0;

	for (let i: i32 = 0; i < capacity; i++) {
		set_node_ptr(i, 0);
		store<u32>(visited_mark_ptr + <usize>i * 4, 0);
	}
	for (let i: i32 = 0; i < MAX_EF; i++) {
		store<u8>(used_mark_ptr + <usize>i, 0);
	}

	// ✅ prevent update_config after init
	mark_index_inited();
}

function get_random_level(): i32 {
	let level: i32 = 0;
	while (rand01() < 0.5 && level < MAX_LAYERS - 1) level++;
	return level;
}

// --- SEARCH LAYER ---
function search_layer(
	query_vec_ptr: usize,
	entry_node: i32,
	layer: i32,
	ef: i32,
	buildMode: bool,
	target: i32,
): i32 {
	if (ef <= 0) return 0;
	if (ef > MAX_EF) {
		ef = MAX_EF;
		ef_clamped_flag = 1;
	}

	// ✅ entry_node 必须是存在节点
	if (entry_node < 0 || entry_node >= max_elements) return 0;
	if (get_node_ptr(entry_node) == 0) return 0;

	visited_next_stamp();

	let cand_cap: i32 = ef * 2 + 32;
	if (cand_cap > MAX_EF) cand_cap = MAX_EF;
	if (cand_cap < ef) cand_cap = ef;

	let cand_size: i32 = 0;
	let res_size: i32 = 0;

	let dist0 = dist_l2_sq(query_vec_ptr, get_vector_ptr(entry_node));

	visited_set(entry_node);
	cand_size = cand_push(cand_size, entry_node, dist0);
	res_size = res_push(res_size, entry_node, dist0);

	// ✅ 当前层 cap
	let cap_this: i32 = layer == 0 ? M_MAX0 : M;

	while (cand_size > 0) {
		cand_size = cand_pop(cand_size);
		let curr_id = tmp_pop_id;
		let curr_dist = tmp_pop_dist;

		if (curr_id < 0) continue;

		if (res_size >= ef && curr_dist > res_peek_worst(res_size)) break;

		let curr_node_ptr = get_node_ptr(curr_id);
		if (curr_node_ptr == 0) continue;

		let runner_ptr = curr_node_ptr + 8;
		for (let l = 0; l < layer; l++) {
			let cap = l == 0 ? M_MAX0 : M;
			runner_ptr += 4 + <usize>cap * 4;
		}

		let neighbor_count = load<i32>(runner_ptr);

		// ✅ clamp neighbor_count，避免坏数据越界读
		if (neighbor_count < 0) continue;
		if (neighbor_count > cap_this) neighbor_count = cap_this;

		let neighbors_start = runner_ptr + 4;

		for (let i = 0; i < neighbor_count; i++) {
			let neighbor_id = load<i32>(neighbors_start + <usize>i * 4);

			if (neighbor_id < 0) continue;
			if (neighbor_id >= max_elements) continue;

			// ✅ skip non-present nodes (sparse-id / corrupted edges)
			if (get_node_ptr(neighbor_id) == 0) continue;

			if (visited_get(neighbor_id)) continue;
			visited_set(neighbor_id);

			let d = dist_l2_sq(query_vec_ptr, get_vector_ptr(neighbor_id));

			if (res_size < ef || d < res_peek_worst(res_size)) {
				if (cand_size < cand_cap) {
					cand_size = cand_push(cand_size, neighbor_id, d);
				}

				if (res_size < ef) {
					res_size = res_push(res_size, neighbor_id, d);
				} else if (d < res_peek_worst(res_size)) {
					res_replace_root(res_size, neighbor_id, d);
				}
			}
		}
	}

	if (!buildMode) {
		return res_to_sorted_full(res_size);
	}

	let pool: i32 = target * 2;
	if (pool < target) pool = target;
	if (pool > ef) pool = ef;
	if (pool > res_size) pool = res_size;

	return res_extract_k_smallest_sorted(res_size, pool);
}

// --- Heuristic neighbor selection ---
function select_neighbors_heuristic(query_vec_ptr: usize, found: i32, target: i32): i32 {
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

// --- LAYER POINTER HELPERS ---
@inline
function get_layer_runner_ptr(node_ptr: usize, layer: i32): usize {
	let runner = node_ptr + 8;
	for (let l = 0; l < layer; l++) {
		let cap = l == 0 ? M_MAX0 : M;
		runner += 4 + <usize>cap * 4;
	}
	return runner; // points to [count] of that layer
}

@inline
function layer_cap(layer: i32): i32 {
	return layer == 0 ? M_MAX0 : M;
}

// --- CONNECTION LOGIC ---
function add_connection(src: i32, dst: i32, layer: i32): void {
	let ptr = get_node_ptr(src);
	if (ptr == 0) return;

	let runner = get_layer_runner_ptr(ptr, layer);

	let count = load<i32>(runner);
	let cap_cur = layer_cap(layer);
	let neighbors_ptr = runner + 4;

	for (let i = 0; i < count; i++) {
		if (load<i32>(neighbors_ptr + <usize>i * 4) == dst) return;
	}

	if (count < cap_cur) {
		store<i32>(neighbors_ptr + <usize>count * 4, dst);
		store<i32>(runner, count + 1);
	} else {
		let src_vec = get_vector_ptr(src);

		// ✅ if there are invalid slots, replace them first
		let worst_idx = -1;
		let worst_dist: f32 = -1.0;

		for (let i = 0; i < count; i++) {
			let n_id = load<i32>(neighbors_ptr + <usize>i * 4);

			if (n_id < 0 || n_id >= max_elements) {
				worst_idx = i;
				worst_dist = 99999999.0;
				break;
			}

			let d = dist_l2_sq(src_vec, get_vector_ptr(n_id));
			if (d > worst_dist) {
				worst_dist = d;
				worst_idx = i;
			}
		}

		let dst_dist = dist_l2_sq(src_vec, get_vector_ptr(dst));
		if (worst_idx >= 0 && dst_dist < worst_dist) {
			// ✅ FIX: remove reverse edge from the victim neighbor
			let victim = load<i32>(neighbors_ptr + <usize>worst_idx * 4);

			store<i32>(neighbors_ptr + <usize>worst_idx * 4, dst);

			if (victim >= 0 && victim < max_elements && victim != dst && victim != src) {
				remove_connection(victim, src, layer);
			}
		}
	}
}

// ✅ remove a connection (used by reconnect to avoid "only add never remove")
function remove_connection(src: i32, dst: i32, layer: i32): void {
	let ptr = get_node_ptr(src);
	if (ptr == 0) return;

	let runner = get_layer_runner_ptr(ptr, layer);
	let count = load<i32>(runner);
	if (count <= 0) return;

	let cap_cur = layer_cap(layer);
	let neighbors_ptr = runner + 4;

	for (let i: i32 = 0; i < count; i++) {
		let nid = load<i32>(neighbors_ptr + <usize>i * 4);
		if (nid != dst) continue;

		// swap with last active
		let last_idx = count - 1;
		let last_id = load<i32>(neighbors_ptr + <usize>last_idx * 4);

		store<i32>(neighbors_ptr + <usize>i * 4, last_id);
		store<i32>(neighbors_ptr + <usize>last_idx * 4, -1);

		store<i32>(runner, count - 1);

		// keep the rest slots as-is
		for (let t = count; t < cap_cur; t++) {
			// no-op
		}
		return;
	}
}

// ✅ overwrite neighbor list for a node at a layer
function overwrite_neighbors(id: i32, layer: i32, picked_count: i32): void {
	let node_ptr = get_node_ptr(id);
	if (node_ptr == 0) return;

	let runner = get_layer_runner_ptr(node_ptr, layer);
	let cap_cur = layer_cap(layer);
	let neighbors_ptr = runner + 4;

	// write new neighbors
	let w: i32 = 0;
	for (let i: i32 = 0; i < picked_count && w < cap_cur; i++) {
		let nid = load<i32>(sel_ids_ptr + <usize>i * 4);
		if (nid < 0 || nid >= max_elements) continue;
		if (nid == id) continue;

		// unique
		let exists = false;
		for (let j: i32 = 0; j < w; j++) {
			if (load<i32>(neighbors_ptr + <usize>j * 4) == nid) {
				exists = true;
				break;
			}
		}
		if (exists) continue;

		store<i32>(neighbors_ptr + <usize>w * 4, nid);
		w++;
	}

	store<i32>(runner, w);

	// clear tail to -1
	for (let i: i32 = w; i < cap_cur; i++) {
		store<i32>(neighbors_ptr + <usize>i * 4, -1);
	}
}

// --- INSERT (id may be new or existing) ---
export function insert(id: i32, vector_data_offset: usize): void {
	if (id < 0 || id >= max_elements) return;

	if (get_node_ptr(id) != 0) {
		update_and_reconnect(id, vector_data_offset);
		return;
	}

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
		for (let j: i32 = 0; j < cap; j++) {
			store<i32>(runner + 4 + <usize>j * 4, -1);
		}
		runner += 4 + <usize>cap * 4;
	}

	set_node_ptr(id, node_ptr);

	if (entry_point_id == -1) {
		entry_point_id = id;
		max_level = level;
		element_count++;
		return;
	}

	let curr_obj = entry_point_id;
	if (curr_obj < 0 || curr_obj >= max_elements) {
		// defensive
		entry_point_id = id;
		max_level = level;
		element_count++;
		return;
	}
	if (get_node_ptr(curr_obj) == 0) {
		// defensive
		entry_point_id = id;
		max_level = level;
		element_count++;
		return;
	}

	let curr_dist = dist_l2_sq(target_vec_ptr, get_vector_ptr(curr_obj));

	// ✅ upper-layer greedy search (defensive clamp)
	for (let l = max_level; l > level; l--) {
		let changed = true;
		while (changed) {
			changed = false;

			let c_ptr = get_node_ptr(curr_obj);
			if (c_ptr == 0) break;

			let r_ptr = c_ptr + 8;
			for (let k = 0; k < l; k++) {
				let cap = k == 0 ? M_MAX0 : M;
				r_ptr += 4 + <usize>cap * 4;
			}

			let cap_layer: i32 = l == 0 ? M_MAX0 : M;
			let count = load<i32>(r_ptr);

			if (count < 0) count = 0;
			if (count > cap_layer) count = cap_layer;

			let n_ptr = r_ptr + 4;

			for (let i = 0; i < count; i++) {
				let neighbor = load<i32>(n_ptr + <usize>i * 4);
				if (neighbor < 0 || neighbor >= max_elements) continue;
				if (get_node_ptr(neighbor) == 0) continue;

				let d = dist_l2_sq(target_vec_ptr, get_vector_ptr(neighbor));
				if (d < curr_dist) {
					curr_dist = d;
					curr_obj = neighbor;
					changed = true;
				}
			}
		}
	}

	for (let l = (level < max_level ? level : max_level); l >= 0; l--) {
		let target = l == 0 ? M_MAX0 : M;

		let found = search_layer(target_vec_ptr, curr_obj, l, EF_CONSTRUCTION, true, target);
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

// --- UPDATE + RECONNECT (FIXED: not "only add", we prune & overwrite) ---
export function update_and_reconnect(id: i32, vector_data_offset: usize): void {
	if (id < 0 || id >= max_elements) return;
	let node_ptr = get_node_ptr(id);
	if (node_ptr == 0) return;

	let target_vec_ptr = get_vector_ptr(id);
	memory.copy(target_vec_ptr, vector_data_offset, get_vector_size());

	let level = load<i32>(node_ptr + 4);
	if (level < 0) return;
	if (level >= MAX_LAYERS) level = MAX_LAYERS - 1;

	if (entry_point_id == -1) return;

	let curr_obj = entry_point_id;
	if (curr_obj < 0 || curr_obj >= max_elements) return;
	if (get_node_ptr(curr_obj) == 0) return;

	let curr_dist = dist_l2_sq(target_vec_ptr, get_vector_ptr(curr_obj));

	// ✅ upper-layer greedy (defensive clamp)
	for (let l = max_level; l > level; l--) {
		let changed = true;
		while (changed) {
			changed = false;

			let c_ptr = get_node_ptr(curr_obj);
			if (c_ptr == 0) break;

			let r_ptr = c_ptr + 8;
			for (let k = 0; k < l; k++) {
				let cap = k == 0 ? M_MAX0 : M;
				r_ptr += 4 + <usize>cap * 4;
			}

			let cap_layer: i32 = l == 0 ? M_MAX0 : M;
			let count = load<i32>(r_ptr);
			if (count < 0) count = 0;
			if (count > cap_layer) count = cap_layer;

			let n_ptr = r_ptr + 4;
			for (let i = 0; i < count; i++) {
				let neighbor = load<i32>(n_ptr + <usize>i * 4);
				if (neighbor < 0 || neighbor >= max_elements) continue;
				if (get_node_ptr(neighbor) == 0) continue;

				let d = dist_l2_sq(target_vec_ptr, get_vector_ptr(neighbor));
				if (d < curr_dist) {
					curr_dist = d;
					curr_obj = neighbor;
					changed = true;
				}
			}
		}
	}

	// --- for each layer: compute new neighbors and overwrite + clean old reverse edges ---
	for (let l = (level < max_level ? level : max_level); l >= 0; l--) {
		let target = l == 0 ? M_MAX0 : M;

		// capture old neighbors
		let old_cap = layer_cap(l);
		let old_ids_count: i32 = 0;
		{
			let runner = get_layer_runner_ptr(node_ptr, l);
			let c = load<i32>(runner);
			if (c < 0) c = 0;
			if (c > old_cap) c = old_cap;

			let n_ptr = runner + 4;
			for (let i: i32 = 0; i < c && i < MAX_EF; i++) {
				let nid = load<i32>(n_ptr + <usize>i * 4);
				store<i32>(old_ids_ptr + <usize>i * 4, nid);
				old_ids_count++;
			}
		}

		let found = search_layer(target_vec_ptr, curr_obj, l, EF_CONSTRUCTION, true, target);
		let picked = select_neighbors_heuristic(target_vec_ptr, found, target);

		for (let i: i32 = 0; i < old_ids_count; i++) {
			let oldN = load<i32>(old_ids_ptr + <usize>i * 4);
			if (oldN < 0 || oldN >= max_elements) continue;
			if (oldN == id) continue;

			let keep = false;
			for (let j: i32 = 0; j < picked; j++) {
				let newN = load<i32>(sel_ids_ptr + <usize>j * 4);
				if (newN == oldN) {
					keep = true;
					break;
				}
			}
			if (!keep) {
				remove_connection(oldN, id, l);
			}
		}

		overwrite_neighbors(id, l, picked);

		for (let i: i32 = 0; i < picked; i++) {
			let nid = load<i32>(sel_ids_ptr + <usize>i * 4);
			if (nid < 0 || nid >= max_elements) continue;
			if (nid == id) continue;
			add_connection(nid, id, l);
		}

		if (found > 0) curr_obj = load<i32>(out_ids_ptr + 0);
	}
}

// --- SEARCH API ---
export function get_results_ptr(): usize {
	return results_buffer_ptr;
}

export function search(query_vec_offset: usize, k: i32): i32 {
	if (entry_point_id == -1) return 0;
	if (max_elements <= 0) return 0;
	if (k <= 0) return 0;

	let curr_obj = entry_point_id;
	if (curr_obj < 0 || curr_obj >= max_elements) return 0;
	if (get_node_ptr(curr_obj) == 0) return 0;

	let curr_dist = dist_l2_sq(query_vec_offset, get_vector_ptr(curr_obj));

	for (let l = max_level; l > 0; l--) {
		let changed = true;
		while (changed) {
			changed = false;
			let c_ptr = get_node_ptr(curr_obj);
			if (c_ptr == 0) break;

			let r_ptr = c_ptr + 8;
			for (let kk = 0; kk < l; kk++) {
				let cap = kk == 0 ? M_MAX0 : M;
				r_ptr += 4 + <usize>cap * 4;
			}

			let cap_layer: i32 = l == 0 ? M_MAX0 : M;
			let count = load<i32>(r_ptr);
			if (count < 0) count = 0;
			if (count > cap_layer) count = cap_layer;

			let n_ptr = r_ptr + 4;

			for (let i = 0; i < count; i++) {
				let neighbor = load<i32>(n_ptr + <usize>i * 4);
				if (neighbor < 0 || neighbor >= max_elements) continue;
				if (get_node_ptr(neighbor) == 0) continue;

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
	if (ef > MAX_EF) {
		ef = MAX_EF;
		ef_clamped_flag = 1;
	}

	let found = search_layer(query_vec_offset, curr_obj, 0, ef, false, 0);

	let out_count = found < k ? found : k;
	if (out_count > results_cap) out_count = results_cap;
	if (out_count > MAX_EF) out_count = MAX_EF;

	for (let i = 0; i < out_count; i++) {
		let id = load<i32>(out_ids_ptr + <usize>i * 4);
		let d = load<f32>(out_dist_ptr + <usize>i * 4);
		store<i32>(results_buffer_ptr + <usize>i * 8, id);
		store<f32>(results_buffer_ptr + <usize>i * 8 + 4, d);
	}

	return out_count;
}

/* ------------------------------
   ✅ Dump V3 (sparse-id safe + M_MAX0 in header)
--------------------------------*/

const MAGIC_HNSW: i32 = 0x57534e48; // "HNSW"
const DUMP_VERSION_V3: i32 = 3;

// ✅ header size is 12 i32 fields = 48 bytes
const HEADER_I32S: usize = 12;
const HEADER_BYTES: usize = HEADER_I32S * 4;

function count_present_nodes(): i32 {
	let n: i32 = 0;
	for (let i: i32 = 0; i < max_elements; i++) {
		if (get_node_ptr(i) != 0) n++;
	}
	return n;
}

export function get_index_dump_size(): usize {
	let present = count_present_nodes();
	let size: usize = 0;

	// ✅ header (12 i32) = 48 bytes
	size += HEADER_BYTES;

	for (let i: i32 = 0; i < max_elements; i++) {
		let node_ptr = get_node_ptr(i);
		if (node_ptr == 0) continue;

		let level = load<i32>(node_ptr + 4);

		size += 4; // id
		size += 4; // level
		size += get_vector_size(); // vector bytes

		for (let l: i32 = 0; l <= level; l++) {
			let cap = l == 0 ? M_MAX0 : M;
			size += 4; // count
			size += <usize>cap * 4; // neighbors full cap
		}
	}

	return size;
}

export function save_index(ptr: usize): usize {
	let off: usize = 0;

	let present = count_present_nodes();

	store<i32>(ptr + off, MAGIC_HNSW); off += 4;
	store<i32>(ptr + off, DUMP_VERSION_V3); off += 4;

	store<i32>(ptr + off, DIM); off += 4;
	store<i32>(ptr + off, M); off += 4;
	store<i32>(ptr + off, M_MAX0); off += 4;
	store<i32>(ptr + off, EF_CONSTRUCTION); off += 4;
	store<i32>(ptr + off, MAX_LAYERS); off += 4;

	store<i32>(ptr + off, max_elements); off += 4;
	store<i32>(ptr + off, present); off += 4;
	store<i32>(ptr + off, entry_point_id); off += 4;
	store<i32>(ptr + off, max_level); off += 4;

	// ✅ results_cap is now always <= MAX_EF, still persisted for host convenience
	store<i32>(ptr + off, results_cap); off += 4;

	for (let i: i32 = 0; i < max_elements; i++) {
		let node_ptr = get_node_ptr(i);
		if (node_ptr == 0) continue;

		let id = load<i32>(node_ptr + 0);
		let level = load<i32>(node_ptr + 4);

		store<i32>(ptr + off, id); off += 4;
		store<i32>(ptr + off, level); off += 4;

		let vptr = get_vector_ptr(id);
		memory.copy(ptr + off, vptr, get_vector_size());
		off += get_vector_size();

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

export function load_index(ptr: usize, size: usize): i32 {
	let off: usize = 0;
	if (size < HEADER_BYTES) return 0;

	let magic = load<i32>(ptr + off); off += 4;
	if (magic != MAGIC_HNSW) return 0;

	let ver = load<i32>(ptr + off); off += 4;
	if (ver != DUMP_VERSION_V3) return 0;

	let dump_dim = load<i32>(ptr + off); off += 4;
	let dump_m = load<i32>(ptr + off); off += 4;
	let dump_mmax0 = load<i32>(ptr + off); off += 4;
	let dump_ef = load<i32>(ptr + off); off += 4;
	let dump_layers = load<i32>(ptr + off); off += 4;

	let dump_max_elements = load<i32>(ptr + off); off += 4;
	let dump_present = load<i32>(ptr + off); off += 4;
	let dump_entry = load<i32>(ptr + off); off += 4;
	let dump_max_level = load<i32>(ptr + off); off += 4;
	let dump_results_cap = load<i32>(ptr + off); off += 4;

	if (
		dump_dim != DIM ||
		dump_m != M ||
		dump_mmax0 != M_MAX0 ||
		dump_ef != EF_CONSTRUCTION ||
		dump_layers != MAX_LAYERS
	) {
		return 0;
	}

	if (dump_max_elements <= 0) return 0;
	if (dump_present < 0) return 0;
	// ✅ present 不能超过 max_elements
	if (dump_present > dump_max_elements) return 0;

	// ✅ entry/max_level 合法性
	if (dump_entry != -1) {
		if (dump_entry < 0 || dump_entry >= dump_max_elements) return 0;
	}
	if (dump_max_level < -1 || dump_max_level >= MAX_LAYERS) return 0;

	// ✅ present>0 时必须有 entry
	if (dump_present > 0 && dump_entry == -1) return 0;

	// ✅ results_cap persisted but clamped
	if (dump_results_cap > 0) {
		results_cap = dump_results_cap;
		if (results_cap > MAX_EF) results_cap = MAX_EF;
	}

	init_index(dump_max_elements);

	entry_point_id = dump_entry;
	max_level = dump_max_level;

	let loaded_count: i32 = 0;

	for (let n: i32 = 0; n < dump_present; n++) {
		if (off + 8 > size) return 0;

		let id = load<i32>(ptr + off); off += 4;
		let level = load<i32>(ptr + off); off += 4;

		if (id < 0 || id >= max_elements) return 0;
		if (level < 0 || level >= MAX_LAYERS) return 0;

		if (off + get_vector_size() > size) return 0;
		memory.copy(get_vector_ptr(id), ptr + off, get_vector_size());
		off += get_vector_size();

		let node_sz = get_node_size(level);
		let node_ptr = alloc(node_sz);

		store<i32>(node_ptr + 0, id);
		store<i32>(node_ptr + 4, level);

		let runner = node_ptr + 8;
		for (let l: i32 = 0; l <= level; l++) {
			let cap = l == 0 ? M_MAX0 : M;
			if (off + 4 + <usize>cap * 4 > size) return 0;

			let count = load<i32>(ptr + off);
			off += 4;

			if (count < 0 || count > cap) return 0;

			store<i32>(runner, count);

			memory.copy(runner + 4, ptr + off, <usize>cap * 4);
			off += <usize>cap * 4;

			// ✅ sanitize neighbors (out-of-range -> -1)
			let neighbors_ptr = runner + 4;
			for (let i: i32 = 0; i < cap; i++) {
				let nid = load<i32>(neighbors_ptr + <usize>i * 4);
				if (nid < 0 || nid >= max_elements) {
					store<i32>(neighbors_ptr + <usize>i * 4, -1);
				}
			}

			runner += 4 + <usize>cap * 4;
		}

		set_node_ptr(id, node_ptr);
		loaded_count++;
	}

	element_count = loaded_count;

	// ✅ entry 必须存在
	if (entry_point_id != -1 && get_node_ptr(entry_point_id) == 0) {
		entry_point_id = -1;
		max_level = -1;
		element_count = 0;
		return 0;
	}

	return 1;
}
