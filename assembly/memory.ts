// Simple Linear Memory Manager (Slab Allocator)
// Purpose: Manage memory for HNSW graph nodes manually to avoid GC.

// Pointers are just u32 indices into Linear Memory

// Global bump pointer
// We start allocating after the heap base provided by the compiler
// Built-in global provided by AS compiler
// declare const __heap_base: usize; // Removed to prevent accidental import generation

let next_ptr: usize = 0;

export function init_memory(): void {
  // Align to 16 bytes
  // @ts-ignore: __heap_base is global
  next_ptr = (__heap_base + 15) & ~15;
}

// Allocate a block of size 'size' bytes
export function alloc(size: usize): usize {
  let ptr = next_ptr;
  // Align next pointer to 4 bytes
  let aligned_size = (size + 3) & ~3;
  let new_ptr = ptr + aligned_size;

  // Auto-grow memory if needed
  let current_pages = memory.size();
  let current_bytes = <usize>current_pages << 16;

  if (new_ptr > current_bytes) {
      let needed_bytes = new_ptr - current_bytes;
      // Calculate pages needed (ceil division)
      let needed_pages = (needed_bytes + 0xFFFF) >>> 16;
      // Grow memory
      memory.grow(<i32>needed_pages);
  }

  next_ptr = new_ptr;
  return ptr;
}

// Get current memory usage
// Returns the offset of the last used byte
export function get_memory_usage(): usize {
  return next_ptr;
}

// Set memory pointer (used during loading)
export function set_memory_usage(ptr: usize): void {
  next_ptr = ptr;
}

// Reset memory (DANGEROUS: Use only for testing or full clear)
export function reset_memory(): void {
  init_memory();
}