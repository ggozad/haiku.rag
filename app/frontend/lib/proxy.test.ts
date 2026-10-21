import { afterEach, describe, expect, it, vi } from "vitest";
import * as backendRoute from "../app/api/backend/[...path]/route";
import * as ingesterRoute from "../app/api/ingester/[...path]/route";
import { proxy } from "./proxy";

type Call = { url: string; init: RequestInit };

function stubFetch(respond: (call: Call) => Promise<Response> | Response) {
	const calls: Call[] = [];
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init: RequestInit) => {
			const call = { url, init };
			calls.push(call);
			return respond(call);
		}),
	);
	return calls;
}

afterEach(() => {
	vi.unstubAllGlobals();
	vi.unstubAllEnvs();
});

describe("proxy", () => {
	it("forwards method, path, query and body, and streams the answer back", async () => {
		const calls = stubFetch(
			() =>
				new Response("streamed", {
					status: 201,
					headers: { "content-type": "text/event-stream", "x-kept": "1" },
				}),
		);
		const request = new Request(
			"http://app/api/backend/v1/chat/stream?a=1&b=two",
			{
				method: "POST",
				body: "payload",
				headers: { host: "app", "content-type": "application/json" },
			},
		);
		const response = await proxy(request, "http://backend:8000/", [
			"v1",
			"chat",
			"stream",
		]);

		expect(calls[0].url).toBe("http://backend:8000/v1/chat/stream?a=1&b=two");
		expect(calls[0].init.method).toBe("POST");
		const headers = calls[0].init.headers as Headers;
		expect(headers.get("host")).toBeNull();
		expect(headers.get("content-type")).toBe("application/json");
		expect(await new Response(calls[0].init.body).text()).toBe("payload");
		expect(response.status).toBe(201);
		expect(response.headers.get("x-kept")).toBe("1");
		expect(await response.text()).toBe("streamed");
	});

	it("hands each upstream chunk on before the upstream finishes", async () => {
		let push: (text: string) => void = () => {};
		const upstream = new ReadableStream<Uint8Array>({
			start(controller) {
				push = (text) => controller.enqueue(new TextEncoder().encode(text));
			},
		});
		stubFetch(() => new Response(upstream));
		const response = await proxy(
			new Request("http://app/x"),
			"http://backend:8000",
			["x"],
		);
		const reader = response.body?.getReader();
		if (!reader) throw new Error("no body");
		push("first");
		const first = await reader.read();
		expect(new TextDecoder().decode(first.value)).toBe("first");
		expect(first.done).toBe(false);
		await reader.cancel();
	});

	it("sends no body on a GET", async () => {
		const calls = stubFetch(() => new Response("{}"));
		await proxy(new Request("http://app/x"), "http://backend:8000", ["x"]);
		expect(calls[0].init.body).toBeUndefined();
	});

	it("encodes each path segment once", async () => {
		const calls = stubFetch(() => new Response("{}"));
		await proxy(new Request("http://app/x"), "http://backend:8000", [
			"api",
			"visualize",
			"a,b c",
		]);
		expect(calls[0].url).toBe("http://backend:8000/api/visualize/a%2Cb%20c");
	});

	it("passes the browser's abort on to the upstream request", async () => {
		const calls = stubFetch(() => new Response("{}"));
		const controller = new AbortController();
		await proxy(
			new Request("http://app/x", { signal: controller.signal }),
			"http://backend:8000",
			["x"],
		);
		const upstreamSignal = calls[0].init.signal as AbortSignal;
		expect(upstreamSignal.aborted).toBe(false);
		controller.abort();
		expect(upstreamSignal.aborted).toBe(true);
	});

	it("answers 499 when the browser left before the upstream answered", async () => {
		const controller = new AbortController();
		stubFetch(() => {
			controller.abort();
			throw new DOMException("aborted", "AbortError");
		});
		const response = await proxy(
			new Request("http://app/x", { signal: controller.signal }),
			"http://backend:8000",
			["x"],
		);
		expect(response.status).toBe(499);
	});

	it("names the unreachable upstream", async () => {
		stubFetch(() => {
			throw new TypeError("connection refused");
		});
		const response = await proxy(
			new Request("http://app/x"),
			"http://backend:8000",
			["x"],
		);
		expect(response.status).toBe(502);
		expect(await response.json()).toEqual({
			error: "http://backend:8000 is unreachable: connection refused",
		});
	});
});

describe("routes", () => {
	const context = { params: Promise.resolve({ path: ["api", "databases"] }) };

	it("sends backend requests to BACKEND_URL", async () => {
		vi.stubEnv("BACKEND_URL", "http://localhost:8001");
		const calls = stubFetch(() => new Response("{}"));
		await backendRoute.GET(new Request("http://app/x"), context);
		expect(calls[0].url).toBe("http://localhost:8001/api/databases");
	});

	it("adds the ingester token server-side", async () => {
		vi.stubEnv("INGESTER_URL", "http://localhost:8765");
		vi.stubEnv("INGESTER_TOKEN", "secret");
		const calls = stubFetch(() => new Response("{}"));
		await ingesterRoute.GET(new Request("http://app/x"), {
			params: Promise.resolve({ path: ["health"] }),
		});
		expect(calls[0].url).toBe("http://localhost:8765/health");
		expect((calls[0].init.headers as Headers).get("authorization")).toBe(
			"Bearer secret",
		);
	});

	it("forwards only the operator actions to the ingester", async () => {
		const calls = stubFetch(() => new Response("{}"));
		const send = (method: "POST" | "DELETE", path: string[]) =>
			(method === "POST" ? ingesterRoute.POST : ingesterRoute.DELETE)(
				new Request("http://app/x", { method }),
				{
					params: Promise.resolve({ path }),
				},
			);
		for (const [method, path] of [
			["POST", ["sources", "docs", "refresh"]],
			["POST", ["jobs", "7", "retry"]],
			["POST", ["dlq", "7", "retry"]],
			["DELETE", ["jobs", "7"]],
		] as const) {
			expect((await send(method, [...path])).status).toBe(200);
		}
		for (const [method, path] of [
			["POST", ["jobs"]],
			["POST", ["sources", "docs", "prune"]],
			["DELETE", ["sources", "docs"]],
			["DELETE", ["jobs", "7", "retry"]],
		] as const) {
			expect((await send(method, [...path])).status).toBe(405);
		}
		expect(calls).toHaveLength(4);
	});

	it("sends no token when none is configured", async () => {
		vi.stubEnv("INGESTER_TOKEN", "");
		const calls = stubFetch(() => new Response("{}"));
		await ingesterRoute.GET(new Request("http://app/x"), {
			params: Promise.resolve({ path: ["health"] }),
		});
		expect(calls[0].url).toBe("http://ingester:8765/health");
		expect((calls[0].init.headers as Headers).get("authorization")).toBeNull();
	});
});
