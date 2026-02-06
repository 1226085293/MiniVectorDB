// src/api/server.ts
import Fastify, { FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import path from "path";
import { MiniVectorDB } from "../index";
import dotenv from "dotenv";

dotenv.config({ path: path.join(__dirname, "../../.env") });

const server: FastifyInstance = Fastify({
	logger: true,
	// ✅ 更容易关干净（避免 keep-alive 连接拖着不退出）
	keepAliveTimeout: 1_000,
	connectionTimeout: 1_000,
});

const db = new MiniVectorDB({
	dim: Number(process.env.VECTOR_DIM) || 384,
	m: Number(process.env.HNSW_M) || 16,
	ef_construction: Number(process.env.HNSW_EF) || 100,
	ef_search: Number(process.env.HNSW_EF_SEARCH) || 50,
	capacity: Number(process.env.HNSW_CAPACITY) || 1_200_000,
	resultsCap: Number(process.env.HNSW_RESULTS_CAP) || 1000,
});

const DUMP_PATH = path.join(__dirname, "../../data/dump.bin");

interface InsertBody {
	id: string;
	vector: number[];
	metadata?: any;
}
interface SearchBody {
	vector: number[];
	k?: number;
	filter?: any;
}

// ✅ 幂等 shutdown（防止重复触发）
let shuttingDown = false;
async function shutdownAndExit(exitCode = 0) {
	if (shuttingDown) return;
	shuttingDown = true;

	// ✅ 兜底：如果还有 handle 卡住，强制退出（unref 避免它本身成为 handle）
	const hardTimer = setTimeout(() => {
		try {
			// eslint-disable-next-line no-console
			console.error("[shutdown] hard timeout reached, forcing exit");
		} catch {}
		process.exit(exitCode || 1);
	}, 6000);
	// @ts-ignore
	hardTimer.unref?.();

	try {
		// 1) 关闭 DB（里面会 close Loki + vector fd）
		await db.close();
	} catch (e) {
		// ignore
	}

	try {
		// 2) 关闭 Fastify（停止接新连接 + 等现有连接结束）
		await server.close();
	} catch (e) {
		// ignore
	}

	// 3) 下一轮事件循环退出（让日志 flush 一下）
	setImmediate(() => process.exit(exitCode));
}

async function start() {
	await server.register(cors, { origin: true });

	// ✅ onClose hook：即使不是 /shutdown，而是进程信号退出，也会走这里
	server.addHook("onClose", async () => {
		try {
			await db.close();
		} catch {}
	});

	await db.init();

	try {
		await db.load(DUMP_PATH);
		server.log.info("Data loaded from dump.");
	} catch {
		server.log.info("No dump found or load failed, starting fresh.");
	}

	server.post<{ Body: InsertBody }>("/insert", async (request, reply) => {
		const { id, vector, metadata } = request.body;

		if (!id || !vector || vector.length !== db.config.dim) {
			return reply
				.code(400)
				.send({ error: `Invalid input. Vector must be ${db.config.dim} dim.` });
		}

		await db.insert({ id, vector, metadata });
		return { status: "ok" };
	});

	server.post<{ Body: SearchBody }>("/search", async (request, reply) => {
		const { vector, k = 10, filter } = request.body;

		if (!vector || vector.length !== db.config.dim) {
			return reply
				.code(400)
				.send({ error: `Invalid vector must be ${db.config.dim} dim.` });
		}

		const f32Vec = new Float32Array(vector);

		// ✅ FIX: k 表示“最终返回数量”，不要在 API 层偷偷 *10
		const results = await db.search(f32Vec, k, filter);
		return { results };
	});

	server.post("/save", async () => {
		await db.save(DUMP_PATH);
		return { status: "saved" };
	});

	server.get("/stats", async () => db.getStats());

	/**
	 * ✅ 测试/CI shutdown：
	 * - 先 reply（避免 fetch 卡住）
	 * - 再异步关库/关服务，并硬超时强退
	 */
	server.post("/shutdown", async (_request, reply) => {
		reply.send({ status: "shutting down" });

		setImmediate(() => {
			shutdownAndExit(0).catch(() => process.exit(1));
		});

		return;
	});

	// 信号退出（CTRL+C、CI 终止）
	process.on("SIGINT", () => shutdownAndExit(0));
	process.on("SIGTERM", () => shutdownAndExit(0));

	const port = parseInt(process.env.API_PORT || "3000", 10);

	try {
		// ✅ 测试建议绑定 127.0.0.1（减少多网卡、双日志、奇怪连接）
		const host = process.env.API_HOST || "127.0.0.1";
		await server.listen({ port, host });
		console.log(`Server listening at http://localhost:${port}`);
	} catch (err) {
		server.log.error(err);
		process.exit(1);
	}
}

start();
