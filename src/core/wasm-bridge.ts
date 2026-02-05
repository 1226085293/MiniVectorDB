import fs from 'fs';
import loader from '@assemblyscript/loader';
import path from 'path';

const WASM_PATH = path.join(__dirname, '../../build/release.wasm');

export class WasmBridge {
  private wasmModule: any;
  private memory!: WebAssembly.Memory;
  private instance!: WebAssembly.Instance;
  
  // Pointers/Offsets
  private inputBufferPtr: number = 0;

  constructor() {
    // We will use the memory exported by WASM
  }

  async init(config?: { dim: number, m: number, ef: number }) {
    const wasmBuffer = fs.readFileSync(WASM_PATH);
    const imports = {
      env: {
        abort: (msg: number, file: number, line: number, col: number) => {
            console.error(`WASM Abort: ${line}:${col}`);
        },
        seed: () => Math.random(), // Required by AS Math.random()
      }
    };

    const result = await loader.instantiate(wasmBuffer, imports);
    this.wasmModule = result.exports;
    this.instance = result.instance;
    // Get the exported memory
    this.memory = this.wasmModule.memory as WebAssembly.Memory;
    
    this.wasmModule.init_memory();

    // Apply Config if provided
    if (config) {
        console.log("Applying WASM Config:", config);
        this.wasmModule.set_config(config.dim, config.m, config.ef);
    }

    // Initialize the index with capacity
    this.wasmModule.init_index(10000);
    
    console.log('WASM Initialized. Memory size:', this.memory.buffer.byteLength);
  }

  // Helper to write a float array to WASM memory
  // Returns the offset
  writeVector(vector: Float32Array): number {
     // In a real app, we would have a dedicated input buffer area in WASM
     // For now, we use a simple alloc from the module if exposed, or just 
     // poke into a known free area. 
     // A better way: Expose an 'alloc_input(size)' function in WASM.
     
     // Let's assume we added 'alloc' to exports in index.ts (we need to do that)
     const ptr = this.wasmModule.alloc(vector.length * 4);
     
     // Create a view on the memory
     const view = new Float32Array(this.memory.buffer, ptr, vector.length);
     view.set(vector);
     return ptr;
  }

  insert(id: number, vector: Float32Array) {
      const ptr = this.writeVector(vector);
      this.wasmModule.insert(id, ptr);
  }

  search(vector: Float32Array, k: number) {
      const ptr = this.writeVector(vector);
      // Now returns count of results found
      const count = this.wasmModule.search(ptr, k);
      
      if (count === 0) return [];

      const resultsPtr = this.wasmModule.get_results_ptr();
      const results = [];
      
      // Read from memory: [id(i32), dist(f32)]
      // We use DataView for safe endianness access, or typed array
      const view = new DataView(this.memory.buffer, resultsPtr, count * 8);
      
      for (let i = 0; i < count; i++) {
          const id = view.getInt32(i * 8, true); // true = little endian (WASM standard)
          const dist = view.getFloat32(i * 8 + 4, true);
          results.push({ id, dist });
      }
      
      return results;
  }

  async save(filepath: string) {
      const usedBytes = this.wasmModule.get_memory_usage();
      console.log(`Saving ${usedBytes} bytes to ${filepath}...`);
      
      // Create a view of the used memory
      const bufferView = new Uint8Array(this.memory.buffer, 0, usedBytes);
      await fs.promises.writeFile(filepath, bufferView);
      console.log("Saved.");
  }

  async load(filepath: string) {
      if (!fs.existsSync(filepath)) {
          console.log("No save file found.");
          return;
      }
      
      const buffer = await fs.promises.readFile(filepath);
      console.log(`Loading ${buffer.byteLength} bytes from ${filepath}...`);
      
      // Ensure WASM memory is large enough
      const currentSize = this.memory.buffer.byteLength;
      if (currentSize < buffer.byteLength) {
          const pagesNeeded = Math.ceil((buffer.byteLength - currentSize) / 65536);
          this.memory.grow(pagesNeeded);
      }
      
      // Copy data into WASM memory
      const wasmView = new Uint8Array(this.memory.buffer);
      wasmView.set(buffer);
      
      // Restore the memory allocator pointer
      this.wasmModule.set_memory_usage(buffer.byteLength);
      
      console.log("Loaded.");
  }
}