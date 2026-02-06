// assembly/pqueue.ts
// Heap-based candidate queues for HNSW (GC-minimal, stable performance)

export class MinHeap {
	ids: Int32Array;
	dists: Float32Array;
	size: i32 = 0;
	capacity: i32;

	constructor(capacity: i32) {
		this.capacity = capacity;
		this.ids = new Int32Array(capacity);
		this.dists = new Float32Array(capacity);
	}

	isEmpty(): bool {
		return this.size == 0;
	}

	peekDist(): f32 {
		if (this.size == 0) return 99999999.0;
		return this.dists[0];
	}

	push(id: i32, dist: f32): void {
		// no capacity limit here; caller decides ef
		if (this.size >= this.capacity) return;
		let i = this.size;
		this.size++;

		while (i > 0) {
			let p = (i - 1) >> 1;
			if (dist >= this.dists[p]) break;
			this.ids[i] = this.ids[p];
			this.dists[i] = this.dists[p];
			i = p;
		}
		this.ids[i] = id;
		this.dists[i] = dist;
	}

	pop(): i32 {
		// returns id, caller reads dist via popDist()
		if (this.size == 0) return -1;
		let rootId = this.ids[0];
		let lastIdx = this.size - 1;
		let lastId = this.ids[lastIdx];
		let lastDist = this.dists[lastIdx];
		this.size--;

		if (this.size > 0) {
			let i = 0;
			while (true) {
				let l = (i << 1) + 1;
				if (l >= this.size) break;
				let r = l + 1;
				let c = l;
				if (r < this.size && this.dists[r] < this.dists[l]) c = r;
				if (this.dists[c] >= lastDist) break;
				this.ids[i] = this.ids[c];
				this.dists[i] = this.dists[c];
				i = c;
			}
			this.ids[i] = lastId;
			this.dists[i] = lastDist;
		}

		return rootId;
	}

	popDist(): f32 {
		// read after pop, but we need dist too; this helper is not used
		return 0.0;
	}

	// Safer combined pop: returns id, writes dist into outDistPtr
	popWithDist(outDistPtr: usize): i32 {
		if (this.size == 0) return -1;

		let rootId = this.ids[0];
		let rootDist = this.dists[0];
		store<f32>(outDistPtr, rootDist);

		let lastIdx = this.size - 1;
		let lastId = this.ids[lastIdx];
		let lastDist = this.dists[lastIdx];
		this.size--;

		if (this.size > 0) {
			let i = 0;
			while (true) {
				let l = (i << 1) + 1;
				if (l >= this.size) break;
				let r = l + 1;
				let c = l;
				if (r < this.size && this.dists[r] < this.dists[l]) c = r;
				if (this.dists[c] >= lastDist) break;
				this.ids[i] = this.ids[c];
				this.dists[i] = this.dists[c];
				i = c;
			}
			this.ids[i] = lastId;
			this.dists[i] = lastDist;
		}

		return rootId;
	}
}

export class MaxHeap {
	ids: Int32Array;
	dists: Float32Array;
	size: i32 = 0;
	capacity: i32;

	constructor(capacity: i32) {
		this.capacity = capacity;
		this.ids = new Int32Array(capacity);
		this.dists = new Float32Array(capacity);
	}

	isEmpty(): bool {
		return this.size == 0;
	}

	worstDist(): f32 {
		// root is worst (max)
		if (this.size == 0) return 99999999.0;
		return this.dists[0];
	}

	// push with cap: keep best 'capacity' items (small dist)
	pushKeepBest(id: i32, dist: f32): void {
		if (this.size < this.capacity) {
			this.pushRaw(id, dist);
			return;
		}

		// full: only accept if better than worst
		if (dist >= this.dists[0]) return;

		// replace root (worst) then heapify down
		this.ids[0] = id;
		this.dists[0] = dist;

		let i = 0;
		while (true) {
			let l = (i << 1) + 1;
			if (l >= this.size) break;
			let r = l + 1;
			let c = l;
			if (r < this.size && this.dists[r] > this.dists[l]) c = r;
			if (this.dists[c] <= this.dists[i]) break;

			// swap i <-> c
			let tmpId = this.ids[i];
			let tmpDist = this.dists[i];
			this.ids[i] = this.ids[c];
			this.dists[i] = this.dists[c];
			this.ids[c] = tmpId;
			this.dists[c] = tmpDist;

			i = c;
		}
	}

	private pushRaw(id: i32, dist: f32): void {
		let i = this.size;
		this.size++;

		while (i > 0) {
			let p = (i - 1) >> 1;
			if (dist <= this.dists[p]) break;
			this.ids[i] = this.ids[p];
			this.dists[i] = this.dists[p];
			i = p;
		}
		this.ids[i] = id;
		this.dists[i] = dist;
	}

	// Export results sorted ascending by distance into a linear buffer [id(i32), dist(f32)]*
	// Returns count written.
	exportSortedTo(ptr: usize, k: i32): i32 {
		// We have max-heap of best items; to output sorted ascending, do selection:
		// Copy to temp arrays, then partial sort by simple selection (k small)
		let cnt = this.size < k ? this.size : k;
		// simple O(k^2) on cnt<=ef (<=50~200) is fine; no GC
		// We'll write by repeatedly finding min among heap arrays (not modifying heap)
		// Create a "used" marker via Int8Array (in AS, this allocates but tiny); avoid by writing indexes.
		let used = new Uint8Array(this.size);
		for (let out = 0; out < cnt; out++) {
			let bestIdx: i32 = -1;
			let bestDist: f32 = 99999999.0;
			for (let i = 0; i < this.size; i++) {
				if (used[i] == 1) continue;
				let d = this.dists[i];
				if (d < bestDist) {
					bestDist = d;
					bestIdx = i;
				}
			}
			used[bestIdx] = 1;
			store<i32>(ptr + <usize>out * 8, this.ids[bestIdx]);
			store<f32>(ptr + <usize>out * 8 + 4, this.dists[bestIdx]);
		}
		return cnt;
	}
}
