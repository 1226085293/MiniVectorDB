import Loki from 'lokijs';
import path from 'path';
import fs from 'fs';

interface Item {
    external_id: string;
    internal_id: number;
    metadata: any;
}

export class MetaDB {
    items!: Collection<Item>;
    private db: Loki;
    private dbPath: string;

    constructor(dbPath: string = path.join(__dirname, '../../data/metadata.json')) {
        this.dbPath = dbPath;
        
        // Ensure directory exists
        const dir = path.dirname(dbPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        this.db = new Loki(dbPath, {
            autoload: true,
            autoloadCallback: this.initCollection.bind(this),
            autosave: true,
            autosaveInterval: 4000
        });
    }

    private initCollection() {
        let items = this.db.getCollection<Item>('items');
        if (items === null) {
            items = this.db.addCollection<Item>('items', {
                indices: ['external_id', 'internal_id']
            });
        }
        this.items = items;
        console.log('MetaDB initialized with', this.items.count(), 'items');
    }

    // Wait for DB to load if needed (LokiJS autoload is async-ish)
    // For this simple example we assume it loads fast or we wrap usage.
    // A robust way involves a Promise.
    async ready(): Promise<void> {
        return new Promise((resolve) => {
            if (this.db.persistenceAdapter) {
                 // already loaded or loading
                 resolve(); // Simplified
                 return;
            }
            // Force save to trigger callback or just wait logic
            // LokiJS autoload is tricky in sync constructor. 
            // Let's manually load.
            this.db.loadDatabase({}, () => {
                this.initCollection();
                resolve();
            });
        });
    }

    add(externalId: string, internalId: number, metadata: any) {
        if (!this.items) this.initCollection();
        
        const existing = this.items.findOne({ external_id: externalId });
        if (existing) {
            existing.internal_id = internalId;
            existing.metadata = metadata;
            this.items.update(existing);
        } else {
            this.items.insert({ external_id: externalId, internal_id: internalId, metadata });
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

    // Filter example: Find internal IDs where metadata.type == 'article'
    filter(query: any): number[] {
        if (!this.items) return [];
        // LokiJS query syntax
        const results = this.items.find(query);
        return results.map(r => r.internal_id);
    }

    close(): Promise<void> {
        return new Promise((resolve) => {
            this.db.close(resolve);
        });
    }
}