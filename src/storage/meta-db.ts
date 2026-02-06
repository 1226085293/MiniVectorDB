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

	async ready(): Promise<void> {
		return new Promise((resolve, reject) => {
			this.db.loadDatabase({}, (err) => {
				if (err) return reject(err);
				this.initCollection();
				resolve();
			});
		});
	}

	/**
	 * Bulk mode:
	 * - disables autosave during massive insert/update
	 * - endBulk will do a single saveDatabase()
	 */
	beginBulk(): void {
		this.bulkDepth++;
		if (this.bulkDepth === 1) {
			this.autosaveWasEnabled = true;
			try {
				// @ts-ignore
				this.db.autosaveDisable();
			} catch {
				// ignore
			}
		}
	}

	async endBulk(): Promise<void> {
		if (this.bulkDepth <= 0) return;
		this.bulkDepth--;
		if (this.bulkDepth === 0) {
			await new Promise<void>((resolve, reject) => {
				this.db.saveDatabase((err) => {
					if (err) return reject(err);
					resolve();
				});
			});

			if (this.autosaveWasEnabled) {
				try {
					// @ts-ignore
					this.db.autosaveEnable();
				} catch {
					// ignore
				}
			}
		}
	}

	// ---- Single ops ----
	add(externalId: string, internalId: number, metadata: any) {
		if (!this.items) this.initCollection();

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

	// ---- Batch ops (✅ perf) ----
	getMany(externalIds: string[]): Map<string, Item> {
		const out = new Map<string, Item>();
		if (!this.items || externalIds.length === 0) return out;

		// de-dup
		const unique = Array.from(new Set(externalIds));
		if (unique.length === 0) return out;

		// Loki supports $in
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
			}
		}

		if (toInsert.length > 0) this.items.insert(toInsert as any);
		if (toUpdate.length > 0) this.items.update(toUpdate as any);
	}

	count(): number {
		if (!this.items) return 0;
		return this.items.count();
	}

	filter(query: any): number[] {
		if (!this.items) return [];
		return this.items.find(query).map((r) => r.internal_id);
	}

	close(): Promise<void> {
		return new Promise((resolve) => this.db.close(resolve));
	}
}
