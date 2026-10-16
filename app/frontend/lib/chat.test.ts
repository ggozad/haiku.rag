import type { Message } from "@ag-ui/client";
import { describe, expect, it, vi } from "vitest";
import {
	checkpoint,
	evidenceAnchors,
	evidenceOf,
	latestTurnRuns,
	loadDraft,
	runTurn,
	saveDraft,
	type TurnAgent,
	textOf,
	toolResults,
} from "./chat";

const citation = (chunk_id: string) => ({
	index: 1,
	document_id: "d",
	source: "alpha",
	chunk_id,
	document_uri: "u",
	document_title: "t",
	page_numbers: [],
	headings: null,
	content: "c",
});

describe("evidenceOf", () => {
	it("lists the cited citations in order and the last successful program", () => {
		const state = {
			rag: {
				citations: ["b", "a", "missing"],
				citation_index: { a: citation("a"), b: citation("b") },
				executions: [
					{ code: "first()", success: true },
					{ code: "second()", success: true },
					{ code: "failed()", success: false },
				],
			},
		};
		const evidence = evidenceOf(state);
		expect(evidence.citations.map((c) => c.chunk_id)).toEqual(["b", "a"]);
		expect(evidence.program).toBe("second()");
	});

	it("is empty for a state without the rag namespace", () => {
		expect(evidenceOf(undefined)).toEqual({ citations: [], program: null });
	});
});

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

describe("runs and evidence", () => {
	it("gives the latest turn's messages its run, and anchors evidence after its last answer", () => {
		const messages: Message[] = [
			{ id: "u1", role: "user", content: "q" },
			{ id: "a1", role: "assistant", content: "earlier" },
			{ id: "u2", role: "user", content: "q2" },
			{ id: "a2", role: "assistant", content: "draft" },
			{ id: "t1", role: "tool", toolCallId: "c", content: "r" },
			{ id: "a3", role: "assistant", content: "final" },
		];
		const runs = latestTurnRuns(messages, "run-2");
		expect(runs).toEqual({ a2: "run-2", t1: "run-2", a3: "run-2" });
		expect([...evidenceAnchors(messages, { ...runs, a1: "run-1" })]).toEqual([
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
});
