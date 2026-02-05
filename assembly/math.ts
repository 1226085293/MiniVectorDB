import { DIM } from "./config";

// Calculate L2 Distance Squared (Euclidean distance squared)
// Using SIMD for performance
export function dist_l2_sq(ptr1: usize, ptr2: usize): f32 {
  let sum: v128 = f32x4.splat(0);
  let i: i32 = 0;

  // Loop with SIMD (process 4 floats at a time)
  // Ensure DIM is a multiple of 4 in config
  while (i < DIM) {
    let v1: v128 = v128.load(ptr1 + <usize>(i * 4));
    let v2: v128 = v128.load(ptr2 + <usize>(i * 4));
    
    let diff: v128 = f32x4.sub(v1, v2);
    let sq: v128 = f32x4.mul(diff, diff);
    sum = f32x4.add(sum, sq);
    
    i += 4;
  }

  // Reduce sum: sum.x + sum.y + sum.z + sum.w
  // Using a more portable way for reduction
  // v128 is 128-bit, holding 4 f32s. 
  
  // Extract lanes
  return f32x4.extract_lane(sum, 0) + 
         f32x4.extract_lane(sum, 1) + 
         f32x4.extract_lane(sum, 2) + 
         f32x4.extract_lane(sum, 3);
}

// Calculate Dot Product
// Useful for Cosine Similarity (if vectors are normalized)
export function dist_dot(ptr1: usize, ptr2: usize): f32 {
    let sum: v128 = f32x4.splat(0);
    let i: i32 = 0;
  
    while (i < DIM) {
      let v1: v128 = v128.load(ptr1 + <usize>(i * 4));
      let v2: v128 = v128.load(ptr2 + <usize>(i * 4));
      
      let prod: v128 = f32x4.mul(v1, v2);
      sum = f32x4.add(sum, prod);
      
      i += 4;
    }
    
    return f32x4.extract_lane(sum, 0) + 
           f32x4.extract_lane(sum, 1) + 
           f32x4.extract_lane(sum, 2) + 
           f32x4.extract_lane(sum, 3);
  }

// L2 Distance for Int8 vectors
// Returns i32 squared distance
export function dist_l2_i8(ptr1: usize, ptr2: usize): i32 {
    let sum: v128 = i32x4.splat(0);
    let i: i32 = 0;

    // Process 16 bytes (int8) at a time if DIM is multiple of 16
    // Or 8 bytes if multiple of 8.
    // Assuming DIM is multiple of 16 for best speed.
    // If DIM=128, it fits perfectly.
    
    while (i < DIM) {
        // Load 16 x i8
        let v1 = v128.load(ptr1 + <usize>i);
        let v2 = v128.load(ptr2 + <usize>i);
        
        // Extending sub: (v1 - v2) -> i16
        // WASM SIMD has extmul but not simple ext_sub easily for all lanes without shuffling.
        // Better approach for L2 sq: 
        // 1. AbsDiff: |a - b| (u8)
        // 2. Widen to i16
        // 3. Square
        // 4. Accumulate to i32
        
        // Sadly v128.sum_squares_diff doesn't exist directly.
        // We do: i16x8_extadd_pairwise_i8x16(v1, v2) ... complex.
        
        // Simpler implementation for now (loop unroll or smaller chunks):
        // Let's process 8 at a time to expand to i16 safely.
        
        // Strategy: Load 128-bit, unpack low/high to i16, sub, square, add.
        
        let v1_lo = i16x8.extend_low_i8x16(v1);
        let v1_hi = i16x8.extend_high_i8x16(v1);
        let v2_lo = i16x8.extend_low_i8x16(v2);
        let v2_hi = i16x8.extend_high_i8x16(v2);
        
        let diff_lo = i16x8.sub(v1_lo, v2_lo);
        let diff_hi = i16x8.sub(v1_hi, v2_hi);
        
        let sq_lo = i16x8.mul(diff_lo, diff_lo);
        let sq_hi = i16x8.mul(diff_hi, diff_hi);
        
        // Now we have i16 squares. We need to accumulate to i32 to avoid overflow (128*128*128 > 65535)
        // extend_low/high again from i16 to i32
        
        let sq_lo_lo = i32x4.extend_low_i16x8(sq_lo);
        let sq_lo_hi = i32x4.extend_high_i16x8(sq_lo);
        let sq_hi_lo = i32x4.extend_low_i16x8(sq_hi);
        let sq_hi_hi = i32x4.extend_high_i16x8(sq_hi);
        
        sum = i32x4.add(sum, sq_lo_lo);
        sum = i32x4.add(sum, sq_lo_hi);
        sum = i32x4.add(sum, sq_hi_lo);
        sum = i32x4.add(sum, sq_hi_hi);
        
        i += 16;
    }
    
    return i32x4.extract_lane(sum, 0) + 
           i32x4.extract_lane(sum, 1) + 
           i32x4.extract_lane(sum, 2) + 
           i32x4.extract_lane(sum, 3);
}