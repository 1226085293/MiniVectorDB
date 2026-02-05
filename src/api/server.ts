import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import path from 'path';
import { MiniVectorDB } from '../index';

const server: FastifyInstance = Fastify({ logger: true });

// --- Global State ---
const db = new MiniVectorDB();
const DUMP_PATH = path.join(__dirname, '../../data/dump.bin');

// --- Interfaces ---
interface InsertBody {
    id: string;
    vector: number[];
    metadata?: any;
}

interface SearchBody {
    vector: number[];
    k?: number;
    filter?: any; // MongoDB-style filter for LokiJS
}

// --- Setup ---
async function start() {
    await server.register(cors, { origin: true });

    // 1. Initialize DB
    await db.init();

    // Try to load dump if exists
    try {
        await db.load(DUMP_PATH);
        server.log.info("Data loaded from dump.");
    } catch (e) {
        server.log.info("No dump found or load failed, starting fresh.");
    }

    // --- Routes ---

    // 1. Insert
    server.post<{ Body: InsertBody }>('/insert', async (request, reply) => {
        const { id, vector, metadata } = request.body;

        if (!id || !vector || vector.length !== 128) {
            return reply.code(400).send({ error: "Invalid input. Vector must be 128 dim." });
        }

        await db.insert({ id, vector, metadata });
        return { status: "ok" };
    });

    // 2. Search
    server.post<{ Body: SearchBody }>('/search', async (request, reply) => {
        const { vector, k = 10, filter } = request.body;

        if (!vector || vector.length !== 128) {
            return reply.code(400).send({ error: "Invalid vector" });
        }

        const f32Vec = new Float32Array(vector);

        // 1. If filter exists, pre-filter? 
        // Current WASM implementation doesn't support BitMask filter yet.
        // So we do Post-Filtering for v1.
        // Better strategy: Filter first, get allowed Internal IDs? 
        // But HNSW search needs to know forbidden nodes *during* traversal.
        // For now: Just search more (k * 5), then filter in JS.
        
        const searchK = filter ? k * 10 : k;
        const results = await db.search(f32Vec, k, filter);
        return { results };
    });

    // 3. Save
    server.post('/save', async (request, reply) => {
        await db.save(DUMP_PATH);
        return { status: "saved" };
    });
    
    // 4. Stats
    server.get('/stats', async () => {
        return db.getStats();
    });

    // Start
    const port = parseInt(process.env.PORT || '3000');
    try {
        await server.listen({ port, host: '0.0.0.0' });
        console.log(`Server listening at http://localhost:${port}`);
    } catch (err) {
        server.log.error(err);
        process.exit(1);
    }
}

start();