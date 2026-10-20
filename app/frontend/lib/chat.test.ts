import type { Message } from "@ag-ui/client";
import { describe, expect, it, vi } from "vitest";
import {
	checkpoint,
	evidenceAnchors,
	loadDraft,
	loadDraftImages,
	loadPending,
	runTurn,
	saveDraft,
	saveDraftImages,
	savePending,
	settlePending,
	statusOf,
	type TurnAgent,
	textOf,
	toolResults,
} from "./chat";

describe("messages", () => {
	const messages: Message[] = [
		{ id: "u1", role: "user", content: "first?" },
		{ id: "a1", role: "assistant", content: "first answer" },
		{ id: "u2", role: "user", content: "second?" },
		{
			id: "a2",
			role: "assistant",
			toolCalls: [
				{
					id: "call-1",
					type: "function",
					function: { name: "search", arguments: "{}" },
				},
			],
		},
		{ id: "t1", role: "tool", toolCallId: "call-1", content: "results" },
		{ id: "a3", role: "assistant", content: "second answer" },
		{ id: "a4", role: "assistant", content: "" },
	];

	it("maps tool results to their calls", () => {
		expect(toolResults(messages).get("call-1")).toBe("results");
	});

	it("takes a checkpoint that later changes do not reach", () => {
		const saved = checkpoint(messages, { rag: { citations: [] } });
		messages.push({ id: "u3", role: "user", content: "third?" });
		expect(saved.messages).toHaveLength(7);
		expect(checkpoint([], undefined).state).toEqual({});
	});

	it("reads the text of a user message", () => {
		expect(textOf({ id: "u", role: "user", content: "plain" })).toBe("plain");
		expect(
			textOf({
				id: "u",
				role: "user",
				content: [
					{ type: "text", text: "with " },
					{
						type: "image",
						source: { type: "data", value: "x", mimeType: "image/png" },
					},
					{ type: "text", text: "image" },
				],
			}),
		).toBe("with image");
		expect(textOf({ id: "a", role: "assistant", content: "no" })).toBe("");
	});
});

describe("evidenceAnchors", () => {
	it("anchors each run's evidence after its last answer", () => {
		const messages: Message[] = [
			{ id: "u1", role: "user", content: "q" },
			{ id: "a1", role: "assistant", content: "earlier" },
			{ id: "u2", role: "user", content: "q2" },
			{ id: "a2", role: "assistant", content: "draft" },
			{ id: "t1", role: "tool", toolCallId: "c", content: "r" },
			{ id: "a3", role: "assistant", content: "final" },
		];
		const runs = { a1: "run-1", a2: "run-2", t1: "run-2", a3: "run-2" };
		expect([...evidenceAnchors(messages, runs)]).toEqual([
			[1, "run-1"],
			[5, "run-2"],
		]);
	});
});

class FakeAgent implements TurnAgent {
	messages: Message[] = [{ id: "u0", role: "user", content: "before" }];
	state: unknown = { rag: { citations: ["kept"] } };
	constructor(
		private readonly script: (
			agent: FakeAgent,
			subscriber: Parameters<TurnAgent["runAgent"]>[1],
		) => Promise<void>,
	) {}
	addMessage(message: Message) {
		this.messages = [...this.messages, message];
	}
	setMessages(messages: Message[]) {
		this.messages = messages;
	}
	setState(state: Record<string, unknown>) {
		this.state = state;
	}
	async runAgent(
		_parameters: Parameters<TurnAgent["runAgent"]>[0],
		subscriber: Parameters<TurnAgent["runAgent"]>[1],
	) {
		await this.script(this, subscriber);
	}
}

describe("runTurn", () => {
	it("keeps a finished run", async () => {
		const agent = new FakeAgent(async (a, s) => {
			a.addMessage({ id: "a1", role: "assistant", content: "answer" });
			a.setState({ rag: { citations: ["new"] } });
			s.onRunFinishedEvent();
		});
		expect(await runTurn(agent, "q", "run-1", {})).toEqual({ finished: true });
		expect(agent.messages.map((m) => m.role)).toEqual([
			"user",
			"user",
			"assistant",
		]);
		expect(agent.state).toEqual({ rag: { citations: ["new"] } });
	});

	it("undoes a run that reported an error", async () => {
		const agent = new FakeAgent(async (a, s) => {
			a.addMessage({ id: "a1", role: "assistant", content: "half" });
			a.setState({ rag: { citations: ["new"] } });
			s.onRunErrorEvent({ event: { message: "model down" } });
		});
		expect(await runTurn(agent, "q", "run-1", {})).toEqual({
			finished: false,
			failure: "model down",
		});
		expect(agent.messages.map((m) => m.id)).toEqual(["u0"]);
		expect(agent.state).toEqual({ rag: { citations: ["kept"] } });
	});

	it("undoes a run that ended without finishing", async () => {
		const agent = new FakeAgent(async (a) => {
			a.addMessage({ id: "a1", role: "assistant", content: "half" });
		});
		expect(await runTurn(agent, "q", "run-1", {})).toEqual({
			finished: false,
			failure: null,
		});
		expect(agent.messages.map((m) => m.id)).toEqual(["u0"]);
	});

	it("undoes a run whose transport failed", async () => {
		const agent = new FakeAgent(async () => {
			throw new TypeError("network down");
		});
		expect(await runTurn(agent, "q", "run-1", {})).toEqual({
			finished: false,
			failure: "network down",
		});
	});
});

describe("drafts", () => {
	it("keeps a draft per session", () => {
		const store = new Map<string, string>();
		vi.stubGlobal("sessionStorage", {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => store.set(key, value),
			removeItem: (key: string) => store.delete(key),
		});
		saveDraft("s1", "half a question");
		expect(loadDraft("s1")).toBe("half a question");
		expect(loadDraft("s2")).toBe("");
		saveDraft("s1", "");
		expect(loadDraft("s1")).toBe("");
		vi.stubGlobal("sessionStorage", {
			getItem: () => {
				throw new Error("denied");
			},
			setItem: () => {
				throw new Error("denied");
			},
		});
		expect(loadDraft("s1")).toBe("");
		saveDraft("s1", "x");
		vi.unstubAllGlobals();
	});

	it("keeps a draft's images per session, and says when it cannot", () => {
		const store = new Map<string, string>();
		vi.stubGlobal("sessionStorage", {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => store.set(key, value),
			removeItem: (key: string) => store.delete(key),
		});
		const image = { data: "AAAA", mimeType: "image/png", name: "a.png" };
		expect(saveDraftImages("s1", [image])).toBe(true);
		expect(loadDraftImages("s1")).toEqual([image]);
		expect(loadDraftImages("s2")).toEqual([]);
		expect(saveDraftImages("s1", [])).toBe(true);
		expect(loadDraftImages("s1")).toEqual([]);
		vi.stubGlobal("sessionStorage", {
			getItem: () => {
				throw new Error("denied");
			},
			setItem: () => {
				throw new Error("QuotaExceededError");
			},
			removeItem: () => {
				throw new Error("denied");
			},
		});
		expect(saveDraftImages("s1", [image])).toBe(false);
		expect(loadDraftImages("s1")).toEqual([]);
		vi.unstubAllGlobals();
	});
});

describe("pending runs", () => {
	it("keeps the unsettled run per session until it settles", () => {
		const store = new Map<string, string>();
		vi.stubGlobal("sessionStorage", {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => store.set(key, value),
			removeItem: (key: string) => store.delete(key),
		});
		savePending("s1", { runId: "run-1", text: "q" });
		expect(loadPending("s1")).toEqual({ runId: "run-1", text: "q" });
		expect(loadPending("s2")).toBeNull();
		settlePending("s1", "run-1");
		expect(loadPending("s1")).toBeNull();
		vi.stubGlobal("sessionStorage", {
			getItem: () => {
				throw new Error("denied");
			},
			setItem: () => {
				throw new Error("denied");
			},
		});
		expect(loadPending("s1")).toBeNull();
		savePending("s1", { runId: "run-1", text: "q" });
		settlePending("s1", "run-1");
		vi.unstubAllGlobals();
	});

	it("keeps a run's images, or its text alone when storage refuses them", () => {
		const store = new Map<string, string>();
		let room = Number.POSITIVE_INFINITY;
		vi.stubGlobal("sessionStorage", {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => {
				if (value.length > room) throw new Error("QuotaExceededError");
				store.set(key, value);
			},
			removeItem: (key: string) => store.delete(key),
		});
		const image = { data: "AAAA", mimeType: "image/png", name: "a.png" };
		const run = { runId: "run-1", text: "q", images: [image] };
		savePending("s1", run);
		expect(loadPending("s1")).toEqual(run);
		room = 80;
		savePending("s1", {
			...run,
			images: [{ ...image, data: "A".repeat(100) }],
		});
		expect(loadPending("s1")).toEqual({
			runId: "run-1",
			text: "q",
			images: [],
			imagesDropped: true,
		});
		vi.unstubAllGlobals();
	});

	it("leaves a newer run in place when an older one settles", () => {
		const store = new Map<string, string>();
		vi.stubGlobal("sessionStorage", {
			getItem: (key: string) => store.get(key) ?? null,
			setItem: (key: string, value: string) => store.set(key, value),
			removeItem: (key: string) => store.delete(key),
		});
		savePending("s1", { runId: "run-2", text: "newer" });
		settlePending("s1", "run-1");
		expect(loadPending("s1")).toEqual({ runId: "run-2", text: "newer" });
		vi.unstubAllGlobals();
	});
});

describe("statusOf", () => {
	it("names the grounding, and the breach first", () => {
		expect(statusOf({ citation_status: "grounded" })?.kind).toBe("grounded");
		expect(statusOf({ citation_status: "ungrounded" })?.kind).toBe(
			"ungrounded",
		);
		expect(
			statusOf({ citation_status: "missing", violation: true })?.kind,
		).toBe("breach");
		expect(statusOf({ citation_status: "missing" })).toBeNull();
	});
});
