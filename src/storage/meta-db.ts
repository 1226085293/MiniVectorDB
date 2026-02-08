// src/storage/meta-db.ts
import Loki, { Collection } from "lokijs";
import path from "path";
import fs from "fs";

export interface Item {
	external_id: string;
	internal_id: number;
	metadata: any;

	// ✅ soft-delete flags (kept outside metadata to avoid user-field conflicts)
	deleted?: boolean;
	deleted_at?: number;
}

export class MetaDB {
	items!: Collection<Item>;
	private db: Loki;
	private dbPath: string;

	private bulkDepth = 0;
	private autosaveWasEnabled = true;

	private nextInternalId: number = 0;
	private nextIdInitialized = false;

	// ✅ cached counts (avoid O(n) scans in hot paths)
	private totalCount: number = 0;
	private deletedCount: number = 0;

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
				indices: ["external_id", "internal_id", "deleted"],
			});
		}
		this.items = items;
	}

	private initNextIdAndCountsFromDB() {
		if (!this.items) this.initCollection();

		let maxId = -1;
		let total = 0;
		let deleted = 0;

		const rows = this.items.find();
		for (const r of rows) {
			total++;
			if (typeof r.internal_id === "number" && r.internal_id > maxId)
				maxId = r.internal_id;
			if (r.deleted === true) deleted++;
		}

		this.nextInternalId = maxId + 1;
		this.nextIdInitialized = true;

		this.totalCount = total;
		this.deletedCount = deleted;
	}

	async ready(): Promise<void> {
		return new Promise((resolve, reject) => {
			this.db.loadDatabase({}, (err) => {
				if (err) return reject(err);
				this.initCollection();
				this.initNextIdAndCountsFromDB();
				resolve();
			});
		});
	}

	// ✅ force flush metadata to disk
	saveNow(): Promise<void> {
		if (this.bulkDepth > 0) return Promise.resolve();
		return new Promise<void>((resolve, reject) => {
			this.db.saveDatabase((err) => {
				if (err) return reject(err);
				resolve();
			});
		});
	}

	beginBulk(): void {
		this.bulkDepth++;
		if (this.bulkDepth === 1) {
			this.autosaveWasEnabled = true;
			try {
				// @ts-ignore
				this.db.autosaveDisable();
			} catch {}
		}
	}

	/**
	 * ✅ endBulk(commit)
	 * - commit=true: save DB and restore autosave
	 * - commit=false: rollback in-memory changes by reloading from disk
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
			await new Promise<void>((resolve, reject) => {
				this.db.loadDatabase({}, (err) => {
					if (err) return reject(err);
					this.initCollection();
					this.initNextIdAndCountsFromDB();
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
		if (!this.nextIdInitialized) this.initNextIdAndCountsFromDB();
		const start = this.nextInternalId;
		this.nextInternalId += n;
		return start;
	}

	getNextInternalId(): number {
		if (!this.nextIdInitialized) this.initNextIdAndCountsFromDB();
		return this.nextInternalId;
	}

	getDeletedCount(): number {
		return this.deletedCount;
	}
	getTotalCount(): number {
		return this.totalCount;
	}
	getActiveCount(): number {
		return Math.max(0, this.totalCount - this.deletedCount);
	}

	/**
	 * ✅ all items snapshot (including deleted)
	 * Used by "compact rebuild".
	 */
	allItems(): Item[] {
		if (!this.items) return [];
		return this.items.find();
	}

	add(
		externalId: string,
		internalId: number,
		metadata: any,
		opts?: { deleted?: boolean },
	) {
		if (!this.items) this.initCollection();
		if (!this.nextIdInitialized) this.initNextIdAndCountsFromDB();

		const existing = this.items.findOne({ external_id: externalId });
		const nextDeleted = opts?.deleted === true;

		if (existing) {
			// adjust deleted count if changed
			const prevDeleted = existing.deleted === true;
			if (prevDeleted !== nextDeleted) {
				this.deletedCount += nextDeleted ? 1 : -1;
			}

			existing.internal_id = internalId;
			existing.metadata = metadata;
			existing.deleted = nextDeleted || undefined;
			existing.deleted_at = nextDeleted
				? (existing.deleted_at ?? Date.now())
				: undefined;

			this.items.update(existing);
		} else {
			const newItem: Item = {
				external_id: externalId,
				internal_id: internalId,
				metadata,
				deleted: nextDeleted || undefined,
				deleted_at: nextDeleted ? Date.now() : undefined,
			};
			this.items.insert(newItem);

			this.totalCount += 1;
			if (nextDeleted) this.deletedCount += 1;

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
		entries: {
			external_id: string;
			internal_id: number;
			metadata: any;
			deleted?: boolean;
		}[],
		existingMap?: Map<string, Item>,
	): void {
		if (!this.items) this.initCollection();
		if (!this.nextIdInitialized) this.initNextIdAndCountsFromDB();
		if (entries.length === 0) return;

		const exMap =
			existingMap ?? this.getMany(entries.map((e) => e.external_id));

		const toInsert: Item[] = [];
		const toUpdate: Item[] = [];

		for (const e of entries) {
			const ex = exMap.get(e.external_id);
			const nextDeleted = e.deleted === true;

			if (ex) {
				const prevDeleted = ex.deleted === true;
				if (prevDeleted !== nextDeleted)
					this.deletedCount += nextDeleted ? 1 : -1;

				ex.internal_id = e.internal_id;
				ex.metadata = e.metadata;
				ex.deleted = nextDeleted || undefined;
				ex.deleted_at = nextDeleted ? (ex.deleted_at ?? Date.now()) : undefined;

				toUpdate.push(ex);
			} else {
				toInsert.push({
					external_id: e.external_id,
					internal_id: e.internal_id,
					metadata: e.metadata,
					deleted: nextDeleted || undefined,
					deleted_at: nextDeleted ? Date.now() : undefined,
				});

				this.totalCount += 1;
				if (nextDeleted) this.deletedCount += 1;

				if (e.internal_id >= this.nextInternalId)
					this.nextInternalId = e.internal_id + 1;
			}
		}

		if (toInsert.length > 0) this.items.insert(toInsert as any);
		if (toUpdate.length > 0) this.items.update(toUpdate as any);
	}

	/**
	 * ✅ Soft delete (mark deleted=true)
	 * Returns counts + internalIds changed.
	 */
	markDeletedMany(
		externalIds: string[],
		deleted: boolean = true,
	): {
		changed: number;
		missing: number;
		already: number;
		internalIds: number[];
	} {
		if (!this.items) this.initCollection();
		if (!this.nextIdInitialized) this.initNextIdAndCountsFromDB();

		const unique = Array.from(new Set(externalIds.filter(Boolean)));
		let changed = 0;
		let missing = 0;
		let already = 0;
		const internalIds: number[] = [];

		for (const exId of unique) {
			const item = this.items.findOne({ external_id: exId });
			if (!item) {
				missing++;
				continue;
			}

			const prev = item.deleted === true;
			if (prev === deleted) {
				already++;
				continue;
			}

			item.deleted = deleted || undefined;
			item.deleted_at = deleted ? Date.now() : undefined;
			this.items.update(item);

			this.deletedCount += deleted ? 1 : -1;
			changed++;
			internalIds.push(item.internal_id);
		}

		return { changed, missing, already, internalIds };
	}

	/**
	 * ✅ Update metadata only.
	 * - merge=true: shallow merge (default)
	 * - merge=false: replace
	 */
	updateMetadata(
		externalId: string,
		newMetadata: any,
		opts: { merge?: boolean } = {},
	): boolean {
		if (!this.items) this.initCollection();
		const item = this.items.findOne({ external_id: externalId });
		if (!item) return false;

		const merge = opts.merge !== false;
		item.metadata = merge
			? { ...(item.metadata ?? {}), ...(newMetadata ?? {}) }
			: newMetadata;

		this.items.update(item);
		return true;
	}

	count(): number {
		return this.totalCount;
	}

	/**
	 * ✅ Filtering helper for search
	 * default excludes deleted
	 */
	filterInternalIdSet(
		query: any,
		opts?: { includeDeleted?: boolean },
	): Set<number> {
		const set = new Set<number>();
		if (!this.items) return set;

		const includeDeleted = opts?.includeDeleted === true;

		let q = query;
		if (!includeDeleted) {
			// merge with user query
			if (q && typeof q === "object")
				q = { $and: [q, { deleted: { $ne: true } }] };
			else q = { deleted: { $ne: true } };
		}

		const rows = this.items.find(q as any) as any[];
		for (const r of rows)
			if (typeof r.internal_id === "number") set.add(r.internal_id);
		return set;
	}

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
