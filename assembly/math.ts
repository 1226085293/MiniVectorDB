// assembly/math.ts
import { DIM } from "./config";

/**
 * Int8 L2 distance squared.
 * Portable (no SIMD intrinsics), guaranteed to compile across AS versions.
 * Returns i32 squared distance.
 */
export function dist_l2_i8(ptr1: usize, ptr2: usize): i32 {
	let sum: i32 = 0;
	for (let i: i32 = 0; i < DIM; i++) {
		// load<i8> returns i8, promotes to i32 with sign
		let a: i32 = <i32>load<i8>(ptr1 + <usize>i);
		let b: i32 = <i32>load<i8>(ptr2 + <usize>i);
		let d: i32 = a - b;
		sum += d * d;
	}
	return sum;
}

/**
 * Keep the old API name for callers.
 * In i8 storage mode this returns (f32)dist_l2_i8.
 */
export function dist_l2_sq(ptr1: usize, ptr2: usize): f32 {
	return <f32>dist_l2_i8(ptr1, ptr2);
}

/**
 * Int8 dot product (portable).
 * Returns i32 accumulator.
 */
export function dot_i8(ptr1: usize, ptr2: usize): i32 {
	let sum: i32 = 0;
	for (let i: i32 = 0; i < DIM; i++) {
		let a: i32 = <i32>load<i8>(ptr1 + <usize>i);
		let b: i32 = <i32>load<i8>(ptr2 + <usize>i);
		sum += a * b;
	}
	return sum;
}

/**
 * Keep compatibility.
 * In i8 storage mode this returns (f32)dot_i8.
 */
export function dist_dot(ptr1: usize, ptr2: usize): f32 {
	return <f32>dot_i8(ptr1, ptr2);
}
