// src/core/wasm-bridge.ts
import fs from "fs";
import loader from "@assemblyscript/loader";
import path from "path";

const WASM_PATH = path.join(__dirname, "../../build/release.wasm");

type InitConfig = {
  dim: number;
  m: number;
  ef: number;
  efSearch?: number;
  capacity: number;
  seed?: number;
  resultsCap?: number;
};

export class WasmBridge {
  private wasmModule: any;
  private memory!: WebAssembly.Memory;

  private currentConfig: InitConfig | null = null;

  private scratchI8Ptr = 0;
  private scratchI8Cap = 0;

  private scratchDumpPtr = 0;
  private scratchDumpCap = 0;

  private lock: Promise<void> = Promise.resolve();
  private wasmMaxEf = 0;

  private async withLock<T>(fn: () => T | Promise<T>): Promise<T> {
    let release!: () => void;
    const next = new Promise<void>((r) => (release = r));
    const prev = this.lock;
    this.lock = prev.then(() => next);
    await prev;
    try {
      return await fn();
    } finally {
      release();
    }
  }

  async init(config: InitConfig) {
    const wasmBuffer = fs.readFileSync(WASM_PATH);

    const imports = {
      env: {
        abort: (_msg: number, _file: number, line: number, col: number) => {
          console.error(`WASM Abort: ${line}:${col}`);
        },
        seed: () => Math.random(),
      },
    };

    const result = await loader.instantiate(wasmBuffer, imports);
    this.wasmModule = result.exports;
    this.memory = this.wasmModule.memory as WebAssembly.Memory;

    await this.withLock(async () => {
      this.wasmModule.init_memory();

      if (typeof this.wasmModule.get_max_ef === "function") {
        this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
      }

      this.currentConfig = { ...config };

      this.wasmModule.set_config(config.dim, config.m, config.ef);

      if (typeof this.wasmModule.set_search_config === "function") {
        this.wasmModule.set_search_config((config.efSearch ?? 50) | 0);
      }

      if (typeof this.wasmModule.set_results_cap === "function") {
        const cap = this.clampResultsCap(config.resultsCap ?? 1000);
        this.wasmModule.set_results_cap(cap);
      }

      if (typeof this.wasmModule.seed_rng === "function") {
        const seed = (config.seed != null ? config.seed : (Date.now() ^ (process.pid << 16))) >>> 0;
        this.wasmModule.seed_rng(seed);
      }

      this.wasmModule.init_index(config.capacity);

      this.ensureScratchI8(config.dim);

      this.scratchDumpPtr = 0;
      this.scratchDumpCap = 0;
    });
  }

  private clampResultsCap(cap: number): number {
    const x = cap | 0;
    if (x <= 0) return 1;
    if (this.wasmMaxEf > 0) return Math.min(x, this.wasmMaxEf);
    return x;
  }

  async getMaxEf(): Promise<number> {
    return this.withLock(() => {
      if (this.wasmMaxEf > 0) return this.wasmMaxEf;
      if (typeof this.wasmModule?.get_max_ef === "function") {
        this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
        return this.wasmMaxEf;
      }
      return 0;
    });
  }

  async setEfSearch(efSearch: number): Promise<void> {
    await this.withLock(() => {
      const ef = efSearch | 0;
      if (ef <= 0) return;
      if (typeof this.wasmModule?.set_search_config === "function") {
        this.wasmModule.set_search_config(ef);
      }
    });
  }

  private ensureScratchI8(nBytes: number) {
    if (nBytes <= 0) throw new Error(`Invalid scratch size: ${nBytes}`);
    if (this.scratchI8Ptr !== 0 && this.scratchI8Cap >= nBytes) return;

    const ptr = this.wasmModule.alloc(nBytes);
    this.scratchI8Ptr = ptr;
    this.scratchI8Cap = nBytes;
  }

  private ensureScratchDump(nBytes: number) {
    if (nBytes <= 0) throw new Error(`Invalid dump scratch size: ${nBytes}`);
    if (this.scratchDumpPtr !== 0 && this.scratchDumpCap >= nBytes) return;

    const ptr = this.wasmModule.alloc(nBytes);
    this.scratchDumpPtr = ptr;
    this.scratchDumpCap = nBytes;
  }

  private writeVectorI8Reusable(vector: Int8Array): number {
    this.ensureScratchI8(vector.length);
    const view = new Int8Array(this.memory.buffer, this.scratchI8Ptr, vector.length);
    view.set(vector);
    return this.scratchI8Ptr;
  }

  private writeDumpReusable(buf: Buffer): number {
    this.ensureScratchDump(buf.length);
    new Uint8Array(this.memory.buffer, this.scratchDumpPtr, buf.length).set(buf);
    return this.scratchDumpPtr;
  }

  async getResultsCap(): Promise<number> {
    return this.withLock(() => {
      if (typeof this.wasmModule.get_results_cap === "function") {
        return Number(this.wasmModule.get_results_cap());
      }
      return 0;
    });
  }

  async setResultsCap(cap: number): Promise<void> {
    await this.withLock(() => {
      if (typeof this.wasmModule.set_results_cap === "function") {
        this.wasmModule.set_results_cap(this.clampResultsCap(cap));
      }
    });
  }

  async hasNode(id: number): Promise<boolean> {
    return this.withLock(() => {
      if (typeof this.wasmModule.has_node === "function") return !!this.wasmModule.has_node(id);
      return false;
    });
  }

  async insert(id: number, vectorI8: Int8Array): Promise<void> {
    await this.withLock(() => {
      const ptr = this.writeVectorI8Reusable(vectorI8);
      this.wasmModule.insert(id, ptr);
    });
  }

  async updateVector(id: number, vectorI8: Int8Array): Promise<void> {
    await this.withLock(() => {
      const ptr = this.writeVectorI8Reusable(vectorI8);
      if (typeof this.wasmModule.update_and_reconnect === "function") {
        this.wasmModule.update_and_reconnect(id, ptr);
        return;
      }
      this.wasmModule.update_vector(id, ptr);
    });
  }

  async search(vectorI8: Int8Array, k: number): Promise<{ id: number; dist: number }[]> {
    return this.withLock(() => {
      const ptr = this.writeVectorI8Reusable(vectorI8);
      const count = this.wasmModule.search(ptr, k);
      if (count === 0) return [];

      const resultsPtr = this.wasmModule.get_results_ptr();
      const view = new DataView(this.memory.buffer, resultsPtr, count * 8);

      const results: { id: number; dist: number }[] = [];
      for (let i = 0; i < count; i++) {
        const id = view.getInt32(i * 8, true);
        const dist = view.getFloat32(i * 8 + 4, true);
        results.push({ id, dist });
      }
      return results;
    });
  }

  async save(filePath: string) {
    await this.withLock(async () => {
      const mark =
        typeof this.wasmModule.get_memory_usage === "function"
          ? Number(this.wasmModule.get_memory_usage()) >>> 0
          : 0;

      const dumpSize: number = this.wasmModule.get_index_dump_size();
      const ptr: number = this.wasmModule.alloc(dumpSize);
      const used: number = this.wasmModule.save_index(ptr);

      const view = new Uint8Array(this.memory.buffer, ptr, used);
      await fs.promises.writeFile(filePath, view);

      if (mark && typeof this.wasmModule.set_memory_usage === "function") {
        this.wasmModule.set_memory_usage(mark);
      }
    });
  }

  async load(filePath: string) {
    await this.withLock(async () => {
      if (!fs.existsSync(filePath)) return;

      const buf = await fs.promises.readFile(filePath);

      const reinitFresh = () => {
        if (typeof this.wasmModule.reset_memory === "function") this.wasmModule.reset_memory();
        else this.wasmModule.init_memory();

        this.scratchI8Ptr = 0;
        this.scratchI8Cap = 0;
        this.scratchDumpPtr = 0;
        this.scratchDumpCap = 0;

        if (typeof this.wasmModule.get_max_ef === "function") {
          this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
        }

        const cfg = this.currentConfig;
        if (cfg) {
          this.wasmModule.set_config(cfg.dim, cfg.m, cfg.ef);
          if (typeof this.wasmModule.set_search_config === "function") {
            this.wasmModule.set_search_config((cfg.efSearch ?? 50) | 0);
          }
          if (typeof this.wasmModule.set_results_cap === "function") {
            this.wasmModule.set_results_cap(this.clampResultsCap(cfg.resultsCap ?? 1000));
          }
          if (typeof this.wasmModule.seed_rng === "function") {
            const seed = (cfg.seed != null ? cfg.seed : (Date.now() ^ (process.pid << 16))) >>> 0;
            this.wasmModule.seed_rng(seed);
          }
          this.wasmModule.init_index(cfg.capacity);
          this.ensureScratchI8(cfg.dim);
        }
      };

      try {
        if (typeof this.wasmModule.reset_memory === "function") this.wasmModule.reset_memory();
        else this.wasmModule.init_memory();

        this.scratchI8Ptr = 0;
        this.scratchI8Cap = 0;
        this.scratchDumpPtr = 0;
        this.scratchDumpCap = 0;

        if (typeof this.wasmModule.get_max_ef === "function") {
          this.wasmMaxEf = Number(this.wasmModule.get_max_ef()) | 0;
        }

        // config 允许幂等设置
        const cfg = this.currentConfig;
        if (cfg) {
          this.wasmModule.set_config(cfg.dim, cfg.m, cfg.ef);
          if (typeof this.wasmModule.set_search_config === "function") {
            this.wasmModule.set_search_config((cfg.efSearch ?? 50) | 0);
          }
          if (typeof this.wasmModule.set_results_cap === "function") {
            this.wasmModule.set_results_cap(this.clampResultsCap(cfg.resultsCap ?? 1000));
          }
          if (typeof this.wasmModule.seed_rng === "function") {
            const seed = (cfg.seed != null ? cfg.seed : (Date.now() ^ (process.pid << 16))) >>> 0;
            this.wasmModule.seed_rng(seed);
          }
        }

        const ptr = this.writeDumpReusable(buf);
        const ok: number = this.wasmModule.load_index(ptr, buf.length);
        if (!ok) {
          throw new Error(
            "WASM load_index failed (dump format mismatch / config mismatch / corrupt dump)."
          );
        }

        if (cfg?.dim) this.ensureScratchI8(cfg.dim);
      } catch (e) {
        reinitFresh();
        throw e;
      }
    });
  }
}
