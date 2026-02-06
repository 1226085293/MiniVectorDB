import { dist_l2_sq } from "./math";
import { alloc } from "./memory";
import { M, M_MAX0, EF_CONSTRUCTION, MAX_LAYERS } from "./config";
import { get_node_size, get_vector_size } from "./types";
import { CandidateList, Candidate } from "./pqueue";

// --- GLOBALS ---
let entry_point_id: i32 = -1;
let max_level: i32 = -1;
let element_count: i32 = 0;

let node_offsets_ptr: usize = 0;
let vector_storage_ptr: usize = 0;
let results_buffer_ptr: usize = 0;
let max_elements: i32 = 0;

// --- HELPERS ---
function get_node_ptr(id: i32): usize {
	return load<usize>(node_offsets_ptr + <usize>id * 4);
}
function get_vector_ptr(id: i32): usize {
	return vector_storage_ptr + <usize>id * get_vector_size();
}

export function init_index(capacity: i32): void {
	max_elements = capacity;
	node_offsets_ptr = alloc(<usize>capacity * 4);
	vector_storage_ptr = alloc(<usize>capacity * get_vector_size());
	// Allocate result buffer for max K=1000? 1000 * 8 bytes = 8KB
	// Format: [id: i32, dist: f32] * K
	results_buffer_ptr = alloc(1000 * 8);

	entry_point_id = -1;
	max_level = -1;
	element_count = 0;
}

function get_random_level(): i32 {
	let level: i32 = 0;
	while (Math.random() < 0.5 && level < MAX_LAYERS - 1) {
		level++;
	}
	return level;
}

// --- SEARCH LAYER ---
// Returns closest candidates found in this layer
// Used for both Insertion (finding neighbors) and Search (finding results)
function search_layer(
	query_vec_ptr: usize,
	entry_node: i32,
	layer: i32,
	ef: i32,
): CandidateList {
	let visited = new Set<i32>();
	let candidates = new CandidateList(ef); // Candidates to explore (Min-Heap behavior ideally, here sorted list)
	let results = new CandidateList(ef); // Best results found so far (Max-Heap behavior ideally)

	let dist = dist_l2_sq(query_vec_ptr, get_vector_ptr(entry_node));

	visited.add(entry_node);
	candidates.push(entry_node, dist);
	results.push(entry_node, dist);

	while (!candidates.isEmpty()) {
		let curr = candidates.popClosest(); // Get closest candidate
		if (curr == null) break;

		// Optimization: if closest candidate is worse than worst result, stop?
		// Standard HNSW condition: dist > lowerBound (worst result distance)
		if (curr.distance > results.worstDist()) {
			break;
		}

		let curr_node_ptr = get_node_ptr(curr.id);

		// Move pointer to connections at 'layer'
		let runner_ptr = curr_node_ptr + 8;
		for (let l = 0; l < layer; l++) {
			let count = load<i32>(runner_ptr);
			let cap = l == 0 ? M_MAX0 : M;
			runner_ptr += 4 + <usize>cap * 4;
		}

		let neighbor_count = load<i32>(runner_ptr);
		let neighbors_start = runner_ptr + 4;

		for (let i = 0; i < neighbor_count; i++) {
			let neighbor_id = load<i32>(neighbors_start + <usize>i * 4);
			if (!visited.has(neighbor_id)) {
				visited.add(neighbor_id);
				let d = dist_l2_sq(query_vec_ptr, get_vector_ptr(neighbor_id));

				// Add to results if it fits or is better than worst
				if (d < results.worstDist() || results.size() < ef) {
					candidates.push(neighbor_id, d);
					results.push(neighbor_id, d);
				}
			}
		}
	}
	return results;
}

// --- CONNECTION LOGIC ---
// Connect src to dst at layer. If src is full, prune.
function add_connection(src: i32, dst: i32, layer: i32): void {
	let ptr = get_node_ptr(src);
	let runner = ptr + 8;
	for (let l = 0; l < layer; l++) {
		let c = load<i32>(runner);
		let cap = l == 0 ? M_MAX0 : M;
		runner += 4 + <usize>cap * 4;
	}

	let count = load<i32>(runner);
	let M_cur = layer == 0 ? M_MAX0 : M;
	let neighbors_ptr = runner + 4;

	// Check if already connected (avoid duplicate)
	// Linear scan is ok for small M
	for (let i = 0; i < count; i++) {
		if (load<i32>(neighbors_ptr + <usize>i * 4) == dst) return;
	}

	if (count < M_cur) {
		// Just add
		store<i32>(neighbors_ptr + <usize>count * 4, dst);
		store<i32>(runner, count + 1);
	} else {
		// Full: Select Neighbors (Shrink)
		// Simple strategy: Find worst neighbor, if dst is better, replace.
		let src_vec = get_vector_ptr(src);
		let worst_idx = -1;
		let worst_dist: f32 = -1.0;

		// Find worst in existing
		for (let i = 0; i < count; i++) {
			let n_id = load<i32>(neighbors_ptr + <usize>i * 4);
			let d = dist_l2_sq(src_vec, get_vector_ptr(n_id));
			if (d > worst_dist) {
				worst_dist = d;
				worst_idx = i;
			}
		}

		let dst_dist = dist_l2_sq(src_vec, get_vector_ptr(dst));
		if (dst_dist < worst_dist) {
			// Replace
			store<i32>(neighbors_ptr + <usize>worst_idx * 4, dst);
		}
	}
}

// --- INSERT ---
export function insert(id: i32, vector_data_offset: usize): void {
	// 1. Store Vector
	let target_vec_ptr = get_vector_ptr(id);
	memory.copy(target_vec_ptr, vector_data_offset, get_vector_size());

	let level = get_random_level();

	// 2. Alloc Node
	let node_sz = get_node_size(level);
	let node_ptr = alloc(node_sz);
	store<i32>(node_ptr, id);
	store<i32>(node_ptr + 4, level);
	// Init counts
	let runner = node_ptr + 8;
	for (let l = 0; l <= level; l++) {
		store<i32>(runner, 0);
		let cap = l == 0 ? M_MAX0 : M;
		runner += 4 + <usize>cap * 4;
	}
	store<usize>(node_offsets_ptr + <usize>id * 4, node_ptr);

	// 3. Insert Logic
	if (entry_point_id == -1) {
		entry_point_id = id;
		max_level = level;
		element_count++;
		return;
	}

	let curr_obj = entry_point_id;
	let curr_dist = dist_l2_sq(target_vec_ptr, get_vector_ptr(curr_obj));

	// Phase 1: Search down from max_level to level+1
	for (let l = max_level; l > level; l--) {
		let changed = true;
		while (changed) {
			changed = false;
			// Search immediate neighbors of curr_obj at layer l
			let c_ptr = get_node_ptr(curr_obj);
			let r_ptr = c_ptr + 8;
			for (let k = 0; k < l; k++) {
				let cnt = load<i32>(r_ptr);
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

	// Phase 2: From level down to 0, search and connect
	for (let l = level < max_level ? level : max_level; l >= 0; l--) {
		// Use search_layer to find EF neighbors
		let candidates = search_layer(target_vec_ptr, curr_obj, l, EF_CONSTRUCTION);
		// Select neighbors (heuristic: just top M from candidates)
		let selected_count = 0;
		// Candidates are closest first. Take top M.
		for (let i = 0; i < candidates.size(); i++) {
			let cand = candidates.elements[i];
			add_connection(id, cand.id, l);
			add_connection(cand.id, id, l);
			selected_count++;
			if (selected_count >= M) break;
		}
		// Update curr_obj for next layer to be the closest found here
		if (candidates.size() > 0) {
			let best = candidates.popClosest();
			if (best) curr_obj = best.id;
		}
	}

	if (level > max_level) {
		max_level = level;
		entry_point_id = id;
	}
	element_count++;
}

export function get_results_ptr(): usize {
	return results_buffer_ptr;
}

export function search(query_vec_offset: usize, k: i32): i32 {
	if (entry_point_id == -1) return 0;

	let curr_obj = entry_point_id;
	let curr_dist = dist_l2_sq(query_vec_offset, get_vector_ptr(curr_obj));

	// Phase 1: Zoom in from top layers
	for (let l = max_level; l > 0; l--) {
		let changed = true;
		while (changed) {
			changed = false;
			let c_ptr = get_node_ptr(curr_obj);
			let r_ptr = c_ptr + 8;
			for (let k = 0; k < l; k++) {
				let cnt = load<i32>(r_ptr);
				let cap = k == 0 ? M_MAX0 : M;
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

	// Phase 2: Layer 0 Search with EF (should be > k)
	let ef = k > 50 ? k : 50;
	let results = search_layer(query_vec_offset, curr_obj, 0, ef);

	// Write results to buffer
	// Results in CandidateList are closest-first. We want top K.
	let count = results.size();
	let out_count = count < k ? count : k;

	for (let i = 0; i < out_count; i++) {
		let cand = results.elements[i];
		store<i32>(results_buffer_ptr + <usize>i * 8, cand.id);
		store<f32>(results_buffer_ptr + <usize>i * 8 + 4, cand.distance);
	}

	return out_count;
}

// --- WASM 状态持久化 ---
export function get_state_ptr(): usize {
	// Layout:
	// [0] entry_point_id
	// [1] max_level
	// [2] element_count
	// [3] node_offsets_ptr
	// [4] vector_storage_ptr
	// [5] results_buffer_ptr
	// [6] max_elements
	let ptr = alloc(7 * 4);
	store<i32>(ptr + 0, entry_point_id);
	store<i32>(ptr + 4, max_level);
	store<i32>(ptr + 8, element_count);
	store<usize>(ptr + 12, node_offsets_ptr);
	store<usize>(ptr + 16, vector_storage_ptr);
	store<usize>(ptr + 20, results_buffer_ptr);
	store<i32>(ptr + 24, max_elements);
	return ptr;
}

export function set_state_ptr(ptr: usize): void {
	entry_point_id = load<i32>(ptr + 0);
	max_level = load<i32>(ptr + 4);
	element_count = load<i32>(ptr + 8);
	node_offsets_ptr = load<usize>(ptr + 12);
	vector_storage_ptr = load<usize>(ptr + 16);
	results_buffer_ptr = load<usize>(ptr + 20);
	max_elements = load<i32>(ptr + 24);
}
