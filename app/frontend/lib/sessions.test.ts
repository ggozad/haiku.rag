import { afterEach, describe, expect, it, vi } from "vitest";
import {
	activeSessionId,
	askAbout,
	createSession,
	deleteSession,
	exportUrl,
	getSession,
	listSessions,
	opening,
	reconcile,
	rememberActiveSession,
	startSession,
	updateSession,
} from "./sessions";

function session(overrides: Record<string, unknown> = {}) {
	return {
		id: "s1",
		title: "t",
		messages: [],
		state: {},
		selection: [],
		runs: {},
		message_runs: {},
		inserted: [],
		in_flight_run_id: null,
		...overrides,
	};
}

function stubFetch(answers: unknown[]) {
	const calls: { url: string; init?: RequestInit }[] = [];
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init?: RequestInit) => {
			calls.push({ url, init });
			const answer = answers.shift();
			if (answer instanceof Response) return answer;
			return Response.json(answer);
		}),
	);
	return calls;
}

afterEach(() => {
	vi.unstubAllGlobals();
});

describe("reconcile", () => {
	it("restores nothing for a run the backend stored", async () => {
		stubFetch([session({ runs: { "run-1": {} } })]);
		expect((await reconcile("s1", "run-1", 0)).outcome).toBe("stored");
	});

	it("restores the question for a run the backend did not store", async () => {
		stubFetch([session()]);
		expect((await reconcile("s1", "run-1", 0)).outcome).toBe("not-stored");
	});

	it("counts a stored run although another tab stored a later one", async () => {
		stubFetch([session({ runs: { "run-1": {}, "run-2": {} } })]);
		expect((await reconcile("s1", "run-1", 0)).outcome).toBe("stored");
	});

	it("waits while the commit is still landing", async () => {
		const calls = stubFetch([
			session({ in_flight_run_id: "run-1" }),
			session({ in_flight_run_id: "run-1" }),
			session({ runs: { "run-1": {} } }),
		]);
		expect((await reconcile("s1", "run-1", 0)).outcome).toBe("stored");
		expect(calls).toHaveLength(3);
	});

	it("calls a run still held when the waiting runs out pending", async () => {
		stubFetch([
			session({ in_flight_run_id: "run-1" }),
			session({ in_flight_run_id: "run-1" }),
		]);
		expect((await reconcile("s1", "run-1", 0, 1)).outcome).toBe("pending");
	});

	it("does not wait on another run's hold", async () => {
		stubFetch([session({ in_flight_run_id: "run-9" })]);
		expect((await reconcile("s1", "run-1", 0)).outcome).toBe("not-stored");
	});
});

describe("session requests", () => {
	it("lists, creates, reads, updates and deletes", async () => {
		const calls = stubFetch([
			{ sessions: [{ id: "s1" }] },
			{ id: "s2" },
			session(),
			session({ title: "renamed" }),
			new Response(null, { status: 204 }),
		]);
		expect(await listSessions()).toEqual([{ id: "s1" }]);
		expect((await createSession()).id).toBe("s2");
		expect((await getSession("s1")).id).toBe("s1");
		expect((await updateSession("s1", { title: "renamed" })).title).toBe(
			"renamed",
		);
		await deleteSession("s1");
		expect(calls.map((c) => [c.url, c.init?.method ?? "GET"])).toEqual([
			["/api/backend/api/sessions", "GET"],
			["/api/backend/api/sessions", "POST"],
			["/api/backend/api/sessions/s1", "GET"],
			["/api/backend/api/sessions/s1", "PATCH"],
			["/api/backend/api/sessions/s1", "DELETE"],
		]);
		expect(calls[3].init?.body).toBe(JSON.stringify({ title: "renamed" }));
		expect(exportUrl("a b")).toBe("/api/backend/api/sessions/a%20b/export");
	});

	it("names the backend's refusal", async () => {
		stubFetch([
			Response.json({ error: "a run is in flight" }, { status: 409 }),
			Response.json({ error: "no session" }, { status: 404 }),
		]);
		await expect(updateSession("s1", { title: "x" })).rejects.toThrow(
			"a run is in flight",
		);
		await expect(deleteSession("s1")).rejects.toThrow("no session");
	});
});

describe("active session", () => {
	it("remembers the session per browser, and survives refused storage", () => {
		const store = new Map<string, string>();
		vi.stubGlobal("localStorage", {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => store.set(key, value),
		});
		rememberActiveSession("s1");
		expect(activeSessionId()).toBe("s1");
		vi.stubGlobal("localStorage", {
			getItem: () => {
				throw new Error("denied");
			},
			setItem: () => {
				throw new Error("denied");
			},
		});
		expect(activeSessionId()).toBeNull();
		rememberActiveSession("s1");
	});
});

describe("opening", () => {
	const summaries = [{ id: "new" }, { id: "old" }] as never[];

	it("opens the session the address names, then the remembered, then the newest", () => {
		expect(opening(summaries, "old", "new")).toBe("old");
		expect(opening(summaries, null, "old")).toBe("old");
		expect(opening(summaries, "gone", "gone")).toBe("new");
		expect(opening([], "gone", null)).toBeNull();
	});
});

describe("askAbout", () => {
	it("creates a session selecting the document", async () => {
		const calls: { method: string; url: string; body?: string }[] = [];
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: string, init: RequestInit = {}) => {
				calls.push({
					method: init.method ?? "GET",
					url,
					body: init.body as string,
				});
				return Response.json(
					{ id: "s9" },
					{ status: init.method === "POST" ? 201 : 200 },
				);
			}),
		);
		expect(await askAbout({ source: "orb", id: "d1" })).toBe("s9");
		expect(calls.map((call) => [call.method, call.url])).toEqual([
			["POST", "/api/backend/api/sessions"],
			["PATCH", "/api/backend/api/sessions/s9"],
		]);
		expect(JSON.parse(calls[1].body ?? "")).toEqual({
			selection: [{ source: "orb", id: "d1" }],
		});
	});

	it("deletes the session when the selection fails, and says why", async () => {
		const methods: string[] = [];
		vi.stubGlobal(
			"fetch",
			vi.fn(async (_url: string, init: RequestInit = {}) => {
				methods.push(init.method ?? "GET");
				if (init.method === "PATCH")
					return Response.json({ error: "selection refused" }, { status: 400 });
				if (init.method === "DELETE")
					return new Response(null, { status: 204 });
				return Response.json({ id: "s9" }, { status: 201 });
			}),
		);
		await expect(askAbout({ source: "orb", id: "d1" })).rejects.toThrow(
			"selection refused",
		);
		expect(methods).toEqual(["POST", "PATCH", "DELETE"]);
	});
});

describe("startSession", () => {
	it("loads the session the address names over the remembered one", async () => {
		vi.stubGlobal("localStorage", {
			getItem: () => "old",
			setItem: () => {},
		});
		const read: string[] = [];
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: string) => {
				read.push(url);
				return url.endsWith("/sessions")
					? Response.json({ sessions: [{ id: "new" }, { id: "old" }] })
					: Response.json({ id: url.split("/").pop() });
			}),
		);
		expect((await startSession("new")).id).toBe("new");
		expect(read).toEqual([
			"/api/backend/api/sessions",
			"/api/backend/api/sessions/new",
		]);
	});

	it("creates a session when there is none", async () => {
		vi.stubGlobal("localStorage", { getItem: () => null, setItem: () => {} });
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: string, init: RequestInit = {}) =>
				url.endsWith("/sessions")
					? init.method === "POST"
						? Response.json({ id: "made" }, { status: 201 })
						: Response.json({ sessions: [] })
					: Response.json({ id: url.split("/").pop() }),
			),
		);
		expect((await startSession(null)).id).toBe("made");
	});
});
