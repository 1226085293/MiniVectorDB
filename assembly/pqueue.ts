// A specialized Candidate List for HNSW
// Maintains a sorted list of candidates (closest first)

export class Candidate {
    id: i32;
    distance: f32;
    
    constructor(id: i32, distance: f32) {
        this.id = id;
        this.distance = distance;
    }
}

// A static array wrapper to avoid GC if possible, but for v1 we use class for simplicity.
// We need to manage a list of candidates up to size 'capacity'.

export class CandidateList {
    elements: Candidate[];
    capacity: i32;

    constructor(capacity: i32) {
        this.capacity = capacity;
        this.elements = new Array<Candidate>(0);
    }

    // Add a candidate. Keep sorted (Ascending distance).
    // Returns true if added, false if rejected (worse than worst and full)
    push(id: i32, distance: f32): boolean {
        if (this.elements.length >= this.capacity) {
            // If full, check if new candidate is better than the worst (last)
            let worst = this.elements[this.elements.length - 1];
            if (distance >= worst.distance) {
                return false;
            }
        }

        // Insert in order
        // Binary search for position could be faster, but linear scan is fine for small N
        let inserted = false;
        for (let i = 0; i < this.elements.length; i++) {
            if (distance < this.elements[i].distance) {
                // Manual Insert (Splice replacement)
                // 1. Expand
                this.elements.push(new Candidate(0, 0.0)); // Placeholder
                // 2. Shift right
                for (let k = this.elements.length - 1; k > i; k--) {
                    this.elements[k] = this.elements[k - 1];
                }
                // 3. Set
                this.elements[i] = new Candidate(id, distance);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            this.elements.push(new Candidate(id, distance));
        }

        // Trim
        if (this.elements.length > this.capacity) {
            this.elements.pop();
        }
        return true;
    }

    // Get the closest (first)
    popClosest(): Candidate | null {
        if (this.elements.length == 0) return null;
        return this.elements.shift();
    }

    // Get current worst distance
    worstDist(): f32 {
        if (this.elements.length > 0) {
            return this.elements[this.elements.length - 1].distance;
        }
        return 99999999.0;
    }

    isEmpty(): boolean {
        return this.elements.length == 0;
    }
    
    size(): i32 {
        return this.elements.length;
    }
}