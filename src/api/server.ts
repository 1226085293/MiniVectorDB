// src/api/server.ts
import Fastify, { FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import path from "path";
import dotenv from "dotenv";
import { MiniVectorDB } from "../index";

dotenv.config({ path: path.join(__dirname, "../../.env") });

const server: FastifyInstance = Fastify({
	logger: true,
	keepAliveTimeout: 1_000,
	connectionTimeout: 1_000,
});

type InsertBody = {
	id: string;
	input: any; // string | number[] | ...
	metadata?: any;
};

type SearchBody = {
	query: any; // string | number[] | ...
	topK?: number;
	filter?: any; // Loki query
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
		const { query, topK = 10, filter } = request.body;
		if (query == null) return reply.code(400).send({ error: "Missing query" });

		const results = await db.search(query, { topK, filter });
		return { results };
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
