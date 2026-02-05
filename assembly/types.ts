// Data Structures Layout

import { DIM, M, M_MAX0 } from "./config";

// We handle two main types of objects in memory:
// 1. Vectors (Raw Float32 Arrays)
// 2. Nodes (Graph Structure)

// --- VECTOR STORAGE ---
// Each vector is simply DIM * 4 bytes.
// Converted to function for dynamic configuration
export function get_vector_size(): usize {
    return <usize>DIM * 4;
}

// --- NODE STORAGE ---
// A node in HNSW needs to store:
// - id: i32 (External/Internal ID mapping is handled in JS, here we just store an int ID)
// - level: i32 (Highest level this node exists in)
// - neighbors: Array of pointers (offsets) for each level

// Memory Layout of a Node:
// [0-3]   id: i32
// [4-7]   level: i32
// [8...]  neighbors_data

// Neighbors Data Layout:
// For each level L from 0 to level:
//   [count: i32] (Number of neighbors at this level)
//   [n1, n2, ... nM]: i32 * M (Neighbor IDs)

// Since different nodes have different levels, the size is variable.
// We need a function to calculate size.

export function get_node_size(level: i32): usize {
    // Base size: id + level
    let size: usize = 8;

    // Level 0 has M_MAX0 neighbors
    // Levels 1..level have M neighbors
    
    // Level 0
    size += 4; // count
    size += <usize>M_MAX0 * 4; // neighbors

    // Level 1 to level
    if (level > 0) {
        let num_upper_layers = <usize>level;
        size += num_upper_layers * (4 + <usize>M * 4);
    }

    return size;
}