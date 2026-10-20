import { afterEach, describe, expect, it, vi } from "vitest";
import {
	CYCLE,
	describeFailure,
	formatAge,
	ingester,
	ingestionState,
	Poller,
	type Snapshot,
} from "./ingestion";

type Responder = (path: string, signal: AbortSignal) => Promise<Response>;

function stubIngester(respond: Responder) {
	const calls: string[] = [];
	vi.stubGlobal(
		"fetch",
		vi.fn((url: string, init: RequestInit) => {
			const path = url.replace("/api/ingester/", "");
			calls.push(path);
			return respond(path, init.signal as AbortSignal);
		}),
	);
	return calls;
}

const ok = (path: string) => Promise.resolve(Response.json({ path }));

afterEach(() => {
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

describe("Poller", () => {
	it("reads every panel in one cycle", async () => {
		const calls = stubIngester(ok);
		const snapshots: Snapshot[] = [];
		await new Poller((s) => snapshots.push(s)).cycle();
		expect(calls.sort()).toEqual(Object.values(CYCLE).sort());
		expect(snapshots[0].stats).toEqual({ data: { path: "stats" } });
	});

	it("degrades only the panel whose call failed", async () => {
		stubIngester((path) =>
			path === "providers"
				? Promise.resolve(Response.json({ detail: "boom" }, { status: 500 }))
				: ok(path),
		);
		const snapshots: Snapshot[] = [];
		await new Poller((s) => snapshots.push(s)).cycle();
		expect(snapshots[0].providers).toEqual({ error: "boom" });
		expect(snapshots[0].health.data).toBeDefined();
	});

	it("names the token on a 401", async () => {
		stubIngester(() =>
			Promise.resolve(
				Response.json({ detail: "Unauthorized" }, { status: 401 }),
			),
		);
		const snapshots: Snapshot[] = [];
		await new Poller((s) => snapshots.push(s)).cycle();
		expect(snapshots[0].health.error).toContain("INGESTER_TOKEN");
	});

	it("skips a tick while a cycle is in flight", async () => {
		let release: () => void = () => {};
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		const calls = stubIngester(async (path) => {
			await gate;
			return ok(path);
		});
		const poller = new Poller(() => {});
		const first = poller.cycle();
		await poller.cycle();
		expect(calls).toHaveLength(Object.keys(CYCLE).length);
		release();
		await first;
	});

	it("drops a cycle that a refresh superseded", async () => {
		let release: () => void = () => {};
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		let round = 0;
		stubIngester(async (path, signal) => {
			if (round === 0) {
				await Promise.race([
					gate,
					new Promise((_, reject) =>
						signal.addEventListener("abort", () =>
							reject(new DOMException("aborted", "AbortError")),
						),
					),
				]);
			}
			return Response.json({ round, path });
		});
		const snapshots: Snapshot[] = [];
		const poller = new Poller((s) => snapshots.push(s));
		poller.start();
		round = 1;
		await poller.refresh();
		release();
		await new Promise((resolve) => setTimeout(resolve, 0));
		poller.stop();
		expect(snapshots).toHaveLength(1);
		expect((snapshots[0].stats.data as { round: number }).round).toBe(1);
	});

	it("gives up on a call past the timeout", async () => {
		stubIngester(
			(_path, signal) =>
				new Promise((_, reject) =>
					signal.addEventListener("abort", () =>
						reject(new DOMException("aborted", "AbortError")),
					),
				),
		);
		const snapshots: Snapshot[] = [];
		await new Poller((s) => snapshots.push(s), 3000, 5).cycle();
		expect(snapshots[0].health.error).toBe(
			"The ingester did not answer in time.",
		);
	});

	it("does not read again after it stopped, and an interruption drops the cycle", async () => {
		let release: () => void = () => {};
		const calls = stubIngester(
			(path, signal) =>
				new Promise((resolve, reject) => {
					signal.addEventListener("abort", () => reject(new Error("aborted")));
					release = () => resolve(Response.json({ path }));
				}),
		);
		const snapshots: Snapshot[] = [];
		const poller = new Poller((s) => snapshots.push(s));
		const running = poller.cycle();
		poller.interrupt();
		release();
		await running;
		expect(snapshots).toEqual([]);
		poller.stop();
		const before = calls.length;
		await poller.refresh();
		expect(calls.length).toBe(before);
	});

	it("keeps the time of the last cycle the ingester answered", async () => {
		let up = true;
		stubIngester((path) =>
			up
				? ok(path)
				: Promise.resolve(Response.json({ detail: "down" }, { status: 502 })),
		);
		const snapshots: Snapshot[] = [];
		const poller = new Poller((s) => snapshots.push(s));
		await poller.cycle();
		up = false;
		await poller.cycle();
		expect(snapshots[0].healthyAt).toBe(snapshots[0].at);
		expect(snapshots[1].healthyAt).toBe(snapshots[0].at);
		expect(new Poller(() => {}).healthyAt).toBeNull();
	});

	it("polls on an interval until stopped, and discards the cycle in flight", async () => {
		vi.useFakeTimers();
		const calls = stubIngester(ok);
		const snapshots: Snapshot[] = [];
		const poller = new Poller((s) => snapshots.push(s), 1000);
		poller.start();
		poller.start();
		await vi.advanceTimersByTimeAsync(2500);
		const cycles = calls.length / Object.keys(CYCLE).length;
		expect(cycles).toBe(3);
		poller.stop();
		await vi.advanceTimersByTimeAsync(5000);
		expect(calls.length / Object.keys(CYCLE).length).toBe(3);
	});
});

describe("ingester requests", () => {
	it("gives up on a request past its timeout", async () => {
		stubIngester(
			(_path, signal) =>
				new Promise((_resolve, reject) => {
					if (signal.aborted) reject(new Error("aborted"));
					signal.addEventListener("abort", () => reject(new Error("aborted")));
				}),
		);
		await expect(ingester("database", {}, 5)).rejects.toThrow(
			"The ingester did not answer in time.",
		);
		const caller = new AbortController();
		await expect(
			ingester("jobs", { signal: caller.signal }, 5),
		).rejects.toThrow("The ingester did not answer in time.");
		const left = ingester("jobs", { signal: caller.signal }, 1000);
		caller.abort();
		await expect(left).rejects.toThrow("aborted");
		await expect(
			ingester("jobs", { signal: caller.signal }, 1000),
		).rejects.toThrow("aborted");
	});

	it("returns the parsed answer, or the text when it is not JSON", async () => {
		stubIngester(() => Promise.resolve(new Response("plain")));
		expect(await ingester("config")).toBe("plain");
		stubIngester(() => Promise.resolve(new Response("")));
		expect(await ingester("x", { method: "DELETE" })).toBeNull();
	});

	it("surfaces the ingester's detail on a refusal", async () => {
		stubIngester(() =>
			Promise.resolve(
				Response.json({ detail: "blocking tombstone" }, { status: 409 }),
			),
		);
		await expect(ingester("jobs/1/retry", { method: "POST" })).rejects.toThrow(
			"blocking tombstone",
		);
		expect(describeFailure(502, "")).toBe("The ingester answered 502.");
		stubIngester(() =>
			Promise.resolve(Response.json({ error: "unreachable" }, { status: 502 })),
		);
		await expect(ingester("health")).rejects.toThrow("unreachable");
	});

	it("formats ages", () => {
		expect(formatAge(null)).toBe("—");
		expect(formatAge(42)).toBe("42s");
		expect(formatAge(300)).toBe("5m");
		expect(formatAge(5400)).toBe("1.5h");
	});
});

describe("ingestionState", () => {
	const health = (extra = {}) => ({
		status: "ok",
		queue_counts: { queued: 0, claimed: 0, succeeded: 9, dead: 0 },
		...extra,
	});

	it("names the first state that holds", () => {
		expect(ingestionState(new Error("down"))).toEqual({
			state: "unreachable",
			label: "Ingester unreachable",
		});
		expect(
			ingestionState(
				health({
					status: "degraded",
					queue_counts: { queued: 3, claimed: 1, dead: 2 },
				}),
			).state,
		).toBe("degraded");
		expect(
			ingestionState(
				health({ queue_counts: { queued: 3, claimed: 1, dead: 2 } }),
			),
		).toEqual({ state: "dead", label: "2 dead letters" });
		expect(
			ingestionState(
				health({ queue_counts: { queued: 3, claimed: 1, dead: 0 } }),
			),
		).toEqual({ state: "busy", label: "Ingesting 4" });
		expect(ingestionState(health())).toEqual({
			state: "idle",
			label: "Ingester idle",
		});
	});

	it("counts one dead letter in the singular", () => {
		expect(ingestionState(health({ queue_counts: { dead: 1 } })).label).toBe(
			"1 dead letter",
		);
	});
});
