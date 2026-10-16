import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchDatabases } from "./databases";

afterEach(() => vi.unstubAllGlobals());

describe("fetchDatabases", () => {
	it("asks again after a failure, and keeps a success", async () => {
		const answers = [
			() => Promise.reject(new TypeError("network down")),
			async () => new Response("not json"),
			async () => Response.json({ error: "backend down" }, { status: 503 }),
			async () => Response.json({ vision: true, databases: [] }),
		];
		const fetch = vi.fn(() => (answers.shift() as () => Promise<Response>)());
		vi.stubGlobal("fetch", fetch);
		await expect(fetchDatabases()).rejects.toThrow("network down");
		await expect(fetchDatabases()).rejects.toThrow();
		await expect(fetchDatabases()).rejects.toThrow("backend down");
		expect(await fetchDatabases()).toEqual({ vision: true, databases: [] });
		expect(await fetchDatabases()).toEqual({ vision: true, databases: [] });
		expect(fetch).toHaveBeenCalledTimes(4);
	});
});
