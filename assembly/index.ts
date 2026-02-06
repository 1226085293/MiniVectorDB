// Entry point for the WASM module

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
	search,
	get_results_ptr,
	get_state_ptr,
	set_state_ptr,
} from "./hnsw";
import { dist_l2_sq, dist_dot } from "./math";
import { update_config } from "./config";

export function set_config(dim: i32, m: i32, ef_construction: i32): void {
	update_config(dim, m, ef_construction);
}

// Re-export functions to be used by the host
export { init_memory, alloc, get_memory_usage, set_memory_usage, reset_memory };
export {
	init_index,
	insert,
	search,
	get_results_ptr,
	get_state_ptr,
	set_state_ptr,
};
export { dist_l2_sq, dist_dot };

// Optional: Test function to verify SIMD works
export function test_simd(ptr1: usize, ptr2: usize): f32 {
	return dist_l2_sq(ptr1, ptr2);
}
