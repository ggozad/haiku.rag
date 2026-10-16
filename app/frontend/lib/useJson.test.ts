import { afterEach, describe, expect, it, vi } from "vitest";
import { load } from "./useJson";

function deferred() {
	let resolve: (value: Response) => void = () => {};
	const promise = new Promise<Response>((r) => {
		resolve = r;
	});
	return { promise, resolve };
}

function recorder() {
	const seen: string[] = [];
	return {
		seen,
		on: {
			data: (value: unknown) => seen.push(`data:${JSON.stringify(value)}`),
			error: (message: string) => seen.push(`error:${message}`),
			done: () => seen.push("done"),
		},
	};
}

afterEach(() => vi.unstubAllGlobals());

describe("load", () => {
	it("hands on the body, then finishes", async () => {
		vi.stubGlobal("fetch", async () => Response.json({ a: 1 }));
		const { seen, on } = recorder();
		await load("/x", new AbortController().signal, on);
		expect(seen).toEqual(['data:{"a":1}', "done"]);
	});

	it("names the backend's error", async () => {
		vi.stubGlobal("fetch", async () =>
			Response.json({ error: "bad filter" }, { status: 400 }),
		);
		const { seen, on } = recorder();
		await load("/x", new AbortController().signal, on);
		expect(seen).toEqual(["error:bad filter", "done"]);
	});

	it("lets nothing land once its selection is replaced", async () => {
		const answer = deferred();
		vi.stubGlobal("fetch", () => answer.promise);
		const { seen, on } = recorder();
		const controller = new AbortController();
		const running = load("/x", controller.signal, on);
		controller.abort();
		answer.resolve(Response.json({ a: 1 }));
		await running;
		expect(seen).toEqual([]);
	});

	it("drops a body that finishes parsing after the abort", async () => {
		let send: (text: string) => void = () => {};
		const body = new ReadableStream<Uint8Array>({
			start(stream) {
				send = (text) => {
					stream.enqueue(new TextEncoder().encode(text));
					stream.close();
				};
			},
		});
		vi.stubGlobal("fetch", async () => new Response(body));
		const { seen, on } = recorder();
		const controller = new AbortController();
		const running = load("/x", controller.signal, on);
		await new Promise((resolve) => setTimeout(resolve, 0));
		controller.abort();
		send('{"a":1}');
		await running;
		expect(seen).toEqual([]);
	});

	it("suppresses the error of a replaced request", async () => {
		const controller = new AbortController();
		vi.stubGlobal("fetch", async () => {
			controller.abort();
			throw new DOMException("aborted", "AbortError");
		});
		const { seen, on } = recorder();
		await load("/x", controller.signal, on);
		expect(seen).toEqual([]);
	});
});
