// src/storage/meta-db.ts
import Loki, { Collection } from "lokijs";
import path from "path";
import fs from "fs";

export interface Item {
	external_id: string;
	internal_id: number;
	metadata: any;
}

export class MetaDB {
	items!: Collection<Item>;
	private db: Loki;
	private dbPath: string;

	private bulkDepth = 0;
	private autosaveWasEnabled = true;

	private nextInternalId: number = 0;
	private nextIdInitialized = false;

	constructor(
		dbPath: string = path.join(__dirname, "../../data/metadata.json"),
	) {
		this.dbPath = dbPath;

		const dir = path.dirname(dbPath);
		if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

		this.db = new Loki(dbPath, {
			autosave: true,
			autosaveInterval: 4000,
		});
	}

	getPath(): string {
		return this.dbPath;
	}

	private initCollection() {
		let items = this.db.getCollection<Item>("items");
		if (!items) {
			items = this.db.addCollection<Item>("items", {
				indices: ["external_id", "internal_id"],
			});
		}
		this.items = items;
	}

	private initNextIdFromDB() {
		if (!this.items) this.initCollection();

		let maxId = -1;
		const rows = this.items.find();
		for (const r of rows) {
			if (typeof r.internal_id === "number" && r.internal_id > maxId) {
				maxId = r.internal_id;
			}
		}
		this.nextInternalId = maxId + 1;
		this.nextIdInitialized = true;
	}

	async ready(): Promise<void> {
		return new Promise((resolve, reject) => {
			this.db.loadDatabase({}, (err) => {
				if (err) return reject(err);
				this.initCollection();
				this.initNextIdFromDB();
				resolve();
			});
		});
	}

	beginBulk(): void {
		this.bulkDepth++;
		if (this.bulkDepth === 1) {
			// best-effort: detect autosave enabled (lokijs doesn't expose cleanly in types)
			this.autosaveWasEnabled = true;
			try {
				// @ts-ignore
				this.db.autosaveDisable();
			} catch {}
		}
	}

	/**
	 * ✅ endBulk(commit)
	 * - commit=true: 保存数据库并恢复 autosave
	 * - commit=false: 丢弃 bulk 期间的内存变更（通过 reloadDatabase 回滚到磁盘版本）
	 */
	async endBulk(commit: boolean = true): Promise<void> {
		if (this.bulkDepth <= 0) return;
		this.bulkDepth--;

		if (this.bulkDepth !== 0) return;

		if (commit) {
			await new Promise<void>((resolve, reject) => {
				this.db.saveDatabase((err) => {
					if (err) return reject(err);
					resolve();
				});
			});
		} else {
			// ✅ rollback: reload from disk
			await new Promise<void>((resolve, reject) => {
				this.db.loadDatabase({}, (err) => {
					if (err) return reject(err);
					this.initCollection();
					this.initNextIdFromDB();
					resolve();
				});
			});
		}

		if (this.autosaveWasEnabled) {
			try {
				// @ts-ignore
				this.db.autosaveEnable();
			} catch {}
		}
	}

	allocInternalIds(n: number): number {
		if (n <= 0) throw new Error(`allocInternalIds invalid n=${n}`);
		if (!this.nextIdInitialized) this.initNextIdFromDB();
		const start = this.nextInternalId;
		this.nextInternalId += n;
		return start;
	}

	add(externalId: string, internalId: number, metadata: any) {
		if (!this.items) this.initCollection();
		if (!this.nextIdInitialized) this.initNextIdFromDB();

		const existing = this.items.findOne({ external_id: externalId });
		if (existing) {
			existing.internal_id = internalId;
			existing.metadata = metadata;
			this.items.update(existing);
		} else {
			const newItem: Item = {
				external_id: externalId,
				internal_id: internalId,
				metadata,
			};
			this.items.insert(newItem);

			if (internalId >= this.nextInternalId)
				this.nextInternalId = internalId + 1;
		}
	}

	get(externalId: string): Item | null {
		if (!this.items) return null;
		return this.items.findOne({ external_id: externalId });
	}

	getByInternalId(internalId: number): Item | null {
		if (!this.items) return null;
		return this.items.findOne({ internal_id: internalId });
	}

	getMany(externalIds: string[]): Map<string, Item> {
		const out = new Map<string, Item>();
		if (!this.items || externalIds.length === 0) return out;

		const unique = Array.from(new Set(externalIds));
		if (unique.length === 0) return out;

		const rows = this.items.find({
			external_id: { $in: unique } as any,
		} as any);

		for (const r of rows) out.set(r.external_id, r);
		return out;
	}

	addMany(
		entries: { external_id: string; internal_id: number; metadata: any }[],
		existingMap?: Map<string, Item>,
	): void {
		if (!this.items) this.initCollection();
		if (!this.nextIdInitialized) this.initNextIdFromDB();
		if (entries.length === 0) return;

		const exMap =
			existingMap ?? this.getMany(entries.map((e) => e.external_id));

		const toInsert: Item[] = [];
		const toUpdate: Item[] = [];

		for (const e of entries) {
			const ex = exMap.get(e.external_id);
			if (ex) {
				ex.internal_id = e.internal_id;
				ex.metadata = e.metadata;
				toUpdate.push(ex);
			} else {
				toInsert.push({
					external_id: e.external_id,
					internal_id: e.internal_id,
					metadata: e.metadata,
				});
				if (e.internal_id >= this.nextInternalId)
					this.nextInternalId = e.internal_id + 1;
			}
		}

		if (toInsert.length > 0) this.items.insert(toInsert as any);
		if (toUpdate.length > 0) this.items.update(toUpdate as any);
	}

	count(): number {
		if (!this.items) return 0;
		return this.items.count();
	}

	filterInternalIdSet(query: any): Set<number> {
		const set = new Set<number>();
		if (!this.items) return set;

		const rows = this.items.find(query as any) as any[];
		for (const r of rows) {
			if (typeof r.internal_id === "number") set.add(r.internal_id);
		}
		return set;
	}

	// ✅ 关键修复：关闭前禁用 autosave，避免 interval 残留导致进程不退出
	close(): Promise<void> {
		try {
			// @ts-ignore
			this.db.autosaveDisable?.();
		} catch {}

		return new Promise((resolve) => {
			this.db.close(() => resolve());
		});
	}
}
