// assembly/memory.ts
// Simple Linear Memory Manager (Slab Allocator)
// Purpose: Manage memory for HNSW graph nodes manually to avoid GC.

let next_ptr: usize = 0;

@inline
function alignUp(ptr: usize, align: usize): usize {
	// align must be power-of-two
	return (ptr + (align - 1)) & ~(align - 1);
}

export function init_memory(): void {
	// Align to 16 bytes
	// @ts-ignore: __heap_base is global
	next_ptr = alignUp(__heap_base, 16);
}

// Allocate a block of size 'size' bytes (aligned to 16 for SIMD friendliness)
export function alloc(size: usize): usize {
	return alloc_aligned(size, 16);
}

// Allocate with explicit alignment (power-of-two)
export function alloc_aligned(size: usize, align: usize): usize {
	if (align < 4) align = 4;
	if ((align & (align - 1)) != 0) {
		// not power-of-two -> fallback to 16
		align = 16;
	}

	let ptr = alignUp(next_ptr, align);

	// Align size to 4 bytes for safe i32 stores
	let aligned_size = (size + 3) & ~3;
	let new_ptr = ptr + aligned_size;

	// Auto-grow memory if needed
	let current_pages = memory.size();
	let current_bytes = (<usize>current_pages) << 16;

	if (new_ptr > current_bytes) {
		let needed_bytes = new_ptr - current_bytes;
		let needed_pages = (needed_bytes + 0xffff) >>> 16;

		let oldPages = memory.grow(<i32>needed_pages);
		if (oldPages < 0) {
			unreachable();
		}
	}

	next_ptr = new_ptr;
	return ptr;
}

export function get_memory_usage(): usize {
	return next_ptr;
}

export function set_memory_usage(ptr: usize): void {
	next_ptr = ptr;
}

export function reset_memory(): void {
	init_memory();
}
