// @ts-nocheck

import { DIM } from "./config";

/**
 * Portable scalar fallback (always available)
 */
@inline
function dist_l2_i8_scalar(ptr1: usize, ptr2: usize): i32 {
	let sum: i32 = 0;
	for (let i: i32 = 0; i < DIM; i++) {
		let a: i32 = <i32>load<i8>(ptr1 + <usize>i);
		let b: i32 = <i32>load<i8>(ptr2 + <usize>i);
		let d: i32 = a - b;
		sum += d * d;
	}
	return sum;
}

@inline
function dot_i8_scalar(ptr1: usize, ptr2: usize): i32 {
	let sum: i32 = 0;
	for (let i: i32 = 0; i < DIM; i++) {
		let a: i32 = <i32>load<i8>(ptr1 + <usize>i);
		let b: i32 = <i32>load<i8>(ptr2 + <usize>i);
		sum += a * b;
	}
	return sum;
}

/**
 * SIMD implementations (enabled when ASC_FEATURE_SIMD is defined)
 * Uses:
 * - load 16 bytes -> extend to i16x8 (low/high)
 * - diff in i16
 * - i32x4.dot_i16x8_s(diff, diff) accumulates squares into 4 lanes
 * - horizontal add lanes
 */
@inline
function hsum_i32x4(v: v128): i32 {
	return i32x4.extract_lane(v, 0)
		+ i32x4.extract_lane(v, 1)
		+ i32x4.extract_lane(v, 2)
		+ i32x4.extract_lane(v, 3);
}

@inline
function dist_l2_i8_simd(ptr1: usize, ptr2: usize): i32 {
	let sum: i32 = 0;

	// DIM is multiple of 16 in your typical configs (e.g. 384, 512).
	let n: i32 = DIM & ~15;
	for (let i: i32 = 0; i < n; i += 16) {
		let a8 = v128.load(ptr1 + <usize>i);
		let b8 = v128.load(ptr2 + <usize>i);

		let aLo = i16x8.extend_low_i8x16_s(a8);
		let aHi = i16x8.extend_high_i8x16_s(a8);
		let bLo = i16x8.extend_low_i8x16_s(b8);
		let bHi = i16x8.extend_high_i8x16_s(b8);

		let dLo = i16x8.sub(aLo, bLo);
		let dHi = i16x8.sub(aHi, bHi);

		let accLo = i32x4.dot_i16x8_s(dLo, dLo);
		let accHi = i32x4.dot_i16x8_s(dHi, dHi);

		sum += hsum_i32x4(i32x4.add(accLo, accHi));
	}

	// tail (if DIM not multiple of 16)
	for (let i: i32 = n; i < DIM; i++) {
		let a: i32 = <i32>load<i8>(ptr1 + <usize>i);
		let b: i32 = <i32>load<i8>(ptr2 + <usize>i);
		let d: i32 = a - b;
		sum += d * d;
	}

	return sum;
}

@inline
function dot_i8_simd(ptr1: usize, ptr2: usize): i32 {
	let sum: i32 = 0;

	let n: i32 = DIM & ~15;
	for (let i: i32 = 0; i < n; i += 16) {
		let a8 = v128.load(ptr1 + <usize>i);
		let b8 = v128.load(ptr2 + <usize>i);

		let aLo = i16x8.extend_low_i8x16_s(a8);
		let aHi = i16x8.extend_high_i8x16_s(a8);
		let bLo = i16x8.extend_low_i8x16_s(b8);
		let bHi = i16x8.extend_high_i8x16_s(b8);

		let accLo = i32x4.dot_i16x8_s(aLo, bLo);
		let accHi = i32x4.dot_i16x8_s(aHi, bHi);

		sum += hsum_i32x4(i32x4.add(accLo, accHi));
	}

	for (let i: i32 = n; i < DIM; i++) {
		let a: i32 = <i32>load<i8>(ptr1 + <usize>i);
		let b: i32 = <i32>load<i8>(ptr2 + <usize>i);
		sum += a * b;
	}

	return sum;
}

/**
 * Int8 L2 distance squared.
 * Returns i32 squared distance.
 */
export function dist_l2_i8(ptr1: usize, ptr2: usize): i32 {
	// compile-time feature gate
	if (isDefined(ASC_FEATURE_SIMD)) {
		return dist_l2_i8_simd(ptr1, ptr2);
	}
	return dist_l2_i8_scalar(ptr1, ptr2);
}

/**
 * Keep old API name
 */
export function dist_l2_sq(ptr1: usize, ptr2: usize): f32 {
	return <f32>dist_l2_i8(ptr1, ptr2);
}

/**
 * Int8 dot product
 */
export function dot_i8(ptr1: usize, ptr2: usize): i32 {
	if (isDefined(ASC_FEATURE_SIMD)) {
		return dot_i8_simd(ptr1, ptr2);
	}
	return dot_i8_scalar(ptr1, ptr2);
}

/**
 * Keep compatibility
 */
export function dist_dot(ptr1: usize, ptr2: usize): f32 {
	return <f32>dot_i8(ptr1, ptr2);
}
