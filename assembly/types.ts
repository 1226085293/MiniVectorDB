// Data Structures Layout

import { DIM, M, M_MAX0 } from "./config";

// We handle two main types of objects in memory:
// 1. Vectors (Quantized Int8 Arrays in WASM memory for ANN search)
// 2. Nodes (Graph Structure)

// --- VECTOR STORAGE ---
// Each vector is DIM * 1 bytes (Int8).
// Converted to function for dynamic configuration.
export function get_vector_size(): usize {
	return <usize>DIM;
}

// --- NODE STORAGE ---
// A node in HNSW needs to store:
// - id: i32
// - level: i32
// - neighbors: Array of neighbor IDs for each level
//
// Memory Layout of a Node:
// [0-3]   id: i32
// [4-7]   level: i32
// [8...]  neighbors_data
//
// Neighbors Data Layout:
// For each level L from 0 to level:
//   [count: i32]
//   [n1, n2, ... nM]: i32 * cap
//
// Since different nodes have different levels, the size is variable.

export function get_node_size(level: i32): usize {
	// Base size: id + level
	let size: usize = 8;

	// Level 0
	size += 4; // count
	size += <usize>M_MAX0 * 4; // neighbors

	// Levels 1..level
	if (level > 0) {
		let num_upper_layers = <usize>level;
		size += num_upper_layers * (4 + <usize>M * 4);
	}

	return size;
}
