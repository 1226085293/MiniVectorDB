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
			// Loki stores autosave state internally; we treat current config as enabled if autosave exists.
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
			// single flush
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
