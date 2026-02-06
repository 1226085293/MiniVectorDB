// assembly/index.ts
import {
	init_memory,
	alloc,
	get_memory_usage,
	set_memory_usage,
	reset_memory,
} from "./memory";

import {
	init_index,
	insert,
	update_vector,
	update_and_reconnect,
	search,
	get_results_ptr,
	get_results_cap,
	save_index,
	load_index,
	get_index_dump_size,
	has_node,
	seed_rng,
	set_results_cap,
	get_max_ef,
	was_ef_clamped,
	clear_ef_clamped,
	get_max_elements,
} from "./hnsw";

import { dist_l2_sq, dist_dot } from "./math";
import { update_config, update_search_config } from "./config";

export function set_config(dim: i32, m: i32, ef_construction: i32): void {
	update_config(dim, m, ef_construction);
}

export function set_search_config(ef_search: i32): void {
	update_search_config(ef_search);
}

export { init_memory, alloc, get_memory_usage, set_memory_usage, reset_memory };

export {
	set_results_cap,
	get_results_cap,
	get_max_elements,
	get_max_ef,
	was_ef_clamped,
	clear_ef_clamped,
	init_index,
	insert,
	update_vector,
	update_and_reconnect,
	has_node,
	search,
	get_results_ptr,
	save_index,
	load_index,
	get_index_dump_size,
	seed_rng,
};

export { dist_l2_sq, dist_dot };

export function test_simd(ptr1: usize, ptr2: usize): f32 {
	return dist_l2_sq(ptr1, ptr2);
}
