// src/api/server.ts
import Fastify, { FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import path from "path";
import { MiniVectorDB } from "../index";

const server: FastifyInstance = Fastify({
	logger: true,
	keepAliveTimeout: 1_000,
	connectionTimeout: 1_000,
});

type InsertBody = {
	id: string;
	input: any;
	metadata?: any;
};

type SearchBody = {
	query: any;
	topK?: number;
	filter?: any;
	score?: "l2" | "cosine" | "similarity";
};

type SearchManyBody = {
	queries: any[];
	topK?: number;
	filter?: any;
	score?: "l2" | "cosine" | "similarity";
};

type RemoveBody = {
	id?: string;
	ids?: string[];
};

type UpdateMetadataBody = {
	id: string;
	metadata: any;
	merge?: boolean;
};

type RebuildBody = {
	capacity?: number;
	persist?: boolean;
	compact?: boolean; // ✅ NEW
};

const STORAGE_DIR =
	process.env.MINIVECTOR_STORAGE_DIR || path.join(process.cwd(), "data");

let db: MiniVectorDB;

let shuttingDown = false;
async function shutdownAndExit(exitCode = 0) {
	if (shuttingDown) return;
	shuttingDown = true;

	const hardTimer = setTimeout(() => {
		try {
			console.error("[shutdown] hard timeout reached, forcing exit");
		} catch {}
		process.exit(exitCode || 1);
	}, 6000);
	// @ts-ignore
	hardTimer.unref?.();

	try {
		await db?.close();
	} catch {}
	try {
		await server.close();
	} catch {}

	setImmediate(() => process.exit(exitCode));
}

async function start() {
	await server.register(cors, { origin: true });

	db = await MiniVectorDB.open({
		storageDir: STORAGE_DIR,
		collection: process.env.MINIVECTOR_COLLECTION || undefined,
		modelName: process.env.MODEL_NAME,
		mode: (process.env.MINIVECTOR_MODE as any) || "balanced",
		capacity: process.env.HNSW_CAPACITY
			? Number(process.env.HNSW_CAPACITY)
			: undefined,
		preloadVectors: process.env.PRELOAD_VECTORS === "1",
	});

	server.post<{ Body: InsertBody }>("/insert", async (request, reply) => {
		const { id, input, metadata } = request.body;
		if (!id) return reply.code(400).send({ error: "Missing id" });
		await db.insert({ id, input, metadata });
		return { status: "ok" };
	});

	server.post<{ Body: SearchBody }>("/search", async (request, reply) => {
		const { query, topK = 10, filter, score } = request.body;
		if (query == null) return reply.code(400).send({ error: "Missing query" });
		const results = await db.search(query, { topK, filter, score });
		return { results };
	});

	server.post<{ Body: SearchManyBody }>(
		"/searchMany",
		async (request, reply) => {
			const { queries, topK = 10, filter, score } = request.body;
			if (!Array.isArray(queries) || queries.length === 0) {
				return reply.code(400).send({ error: "Missing queries[]" });
			}
			const results = await db.searchMany(queries, { topK, filter, score });
			return { results };
		},
	);

	server.post<{ Body: RemoveBody }>("/remove", async (request, reply) => {
		const ids = request.body.ids || (request.body.id ? [request.body.id] : []);
		if (!ids.length) return reply.code(400).send({ error: "Missing id/ids" });

		const r = await db.removeMany(ids);
		return { status: "ok", ...r };
	});

	server.post<{ Body: UpdateMetadataBody }>(
		"/updateMetadata",
		async (request, reply) => {
			const { id, metadata, merge } = request.body;
			if (!id) return reply.code(400).send({ error: "Missing id" });

			const ok = await db.updateMetadata(id, metadata, { merge });
			return { status: ok ? "ok" : "not_found" };
		},
	);

	server.post<{ Body: RebuildBody }>("/rebuild", async (request) => {
		const { capacity, persist, compact } = request.body || {};
		const r = await db.rebuild({ capacity, persist, compact });
		return { status: "ok", ...r };
	});

	server.post("/save", async () => {
		await db.save();
		return { status: "saved" };
	});

	server.get("/stats", async () => db.getStats());

	server.post("/shutdown", async (_req, reply) => {
		reply.send({ status: "shutting down" });
		setImmediate(() => shutdownAndExit(0).catch(() => process.exit(1)));
	});

	process.on("SIGINT", () => shutdownAndExit(0));
	process.on("SIGTERM", () => shutdownAndExit(0));

	const port = parseInt(process.env.API_PORT || "3000", 10);
	const host = process.env.API_HOST || "127.0.0.1";

	try {
		await server.listen({ port, host });
		console.log(`Server listening at http://localhost:${port}`);
	} catch (err) {
		server.log.error(err);
		process.exit(1);
	}
}

start();
