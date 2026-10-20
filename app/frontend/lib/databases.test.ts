import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type * as Databases from "./databases";

type Module = typeof Databases;
const info = (available: boolean, documents = 1) => ({
	vision: true,
	databases: [{ name: "alpha", available, error: null, documents }],
});

let module: Module;
let page: EventTarget & { hidden: boolean };

beforeEach(async () => {
	vi.resetModules();
	vi.useFakeTimers();
	page = Object.assign(new EventTarget(), { hidden: false });
	vi.stubGlobal("document", page);
	vi.stubGlobal("window", new EventTarget());
	module = await import("./databases");
});

afterEach(() => {
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

function answering(...answers: (() => Promise<Response>)[]) {
	const fetch = vi.fn(() => (answers.shift() as () => Promise<Response>)());
	vi.stubGlobal("fetch", fetch);
	return fetch;
}

describe("refreshDatabases", () => {
	it("updates every subscriber, including on a later change", async () => {
		answering(
			async () => Response.json(info(false)),
			async () => Response.json(info(true, 7)),
			async () => Response.json(info(false)),
		);
		const first = vi.fn();
		const second = vi.fn();
		module.subscribeDatabases(first);
		module.subscribeDatabases(second);
		await module.refreshDatabases();
		expect(module.databasesSnapshot().info).toEqual(info(false));
		await module.refreshDatabases();
		expect(module.databasesSnapshot().info).toEqual(info(true, 7));
		await module.refreshDatabases();
		expect(module.databasesSnapshot().info).toEqual(info(false));
		expect(first).toHaveBeenCalledTimes(second.mock.calls.length);
		expect(first.mock.calls.length).toBeGreaterThanOrEqual(3);
	});

	it("shares one request between concurrent callers", async () => {
		const fetch = answering(async () => Response.json(info(true)));
		await Promise.all([module.refreshDatabases(), module.refreshDatabases()]);
		expect(fetch).toHaveBeenCalledTimes(1);
	});

	it("keeps the last good list through a failure, and clears the error after", async () => {
		answering(
			async () => Response.json(info(true)),
			async () => Response.json({ error: "backend down" }, { status: 503 }),
			async () => new Response("not json"),
			() => Promise.reject(new TypeError("network down")),
			async () => Response.json(info(true, 2)),
		);
		await module.refreshDatabases();
		await module.refreshDatabases();
		expect(module.databasesSnapshot()).toEqual({
			info: info(true),
			error: "backend down",
		});
		await module.refreshDatabases();
		expect(module.databasesSnapshot().error).toBeTruthy();
		await module.refreshDatabases();
		expect(module.databasesSnapshot().error).toBe("network down");
		await module.refreshDatabases();
		expect(module.databasesSnapshot()).toEqual({
			info: info(true, 2),
			error: null,
		});
	});
});

describe("subscribeDatabases", () => {
	it("reads on subscribe, every 30 s while visible, and on return to the page", async () => {
		const fetch = vi.fn(async () => Response.json(info(true)));
		vi.stubGlobal("fetch", fetch);
		const stop = module.subscribeDatabases(() => {});
		const alsoStop = module.subscribeDatabases(() => {});
		await vi.advanceTimersByTimeAsync(0);
		expect(fetch).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(30_000);
		expect(fetch).toHaveBeenCalledTimes(2);

		page.hidden = true;
		await vi.advanceTimersByTimeAsync(60_000);
		expect(fetch).toHaveBeenCalledTimes(2);

		page.hidden = false;
		page.dispatchEvent(new Event("visibilitychange"));
		await vi.advanceTimersByTimeAsync(0);
		expect(fetch).toHaveBeenCalledTimes(3);

		window.dispatchEvent(new Event("focus"));
		await vi.advanceTimersByTimeAsync(0);
		expect(fetch).toHaveBeenCalledTimes(4);

		stop();
		alsoStop();
		await vi.advanceTimersByTimeAsync(120_000);
		window.dispatchEvent(new Event("focus"));
		page.dispatchEvent(new Event("visibilitychange"));
		await vi.advanceTimersByTimeAsync(0);
		expect(fetch).toHaveBeenCalledTimes(4);
	});
});
