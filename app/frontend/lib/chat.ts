import type { Message } from "@ag-ui/client";
import type { Citation, RAGState } from "./sessionStorage";
import { AGUI_STATE_KEY, normalizeRAGState } from "./sessionStorage";

// A selected document: ids repeat only between copies of a database, so the
// database name is part of its identity.
export interface DocumentRef {
	source: string;
	id: string;
}

export interface Evidence {
	citations: Citation[];
	// The last program the model ran successfully for the question, if any.
	program: string | null;
}

interface CodeExecution {
	code: string;
	success?: boolean;
}

export function ragStateOf(state: unknown): RAGState {
	const namespaced = (state as Record<string, unknown> | undefined)?.[
		AGUI_STATE_KEY
	] as Partial<RAGState> | undefined;
	return normalizeRAGState(namespaced);
}

// What the last question rests on: the state after a run holds that
// question's citations and executions only.
export function evidenceOf(state: unknown): Evidence {
	const rag = ragStateOf(state);
	const citations = rag.citations
		.map((id) => rag.citation_index[id])
		.filter((citation): citation is Citation => citation !== undefined);
	const executions = (rag.executions as CodeExecution[] | undefined) ?? [];
	const successful = executions.filter((entry) => entry.success !== false);
	return {
		citations,
		program: successful.length ? successful[successful.length - 1].code : null,
	};
}

// Tool results by the id of the call they answer.
export function toolResults(messages: Message[]): Map<string, string> {
	const results = new Map<string, string>();
	for (const message of messages) {
		if (message.role === "tool") {
			results.set(message.toolCallId, message.content);
		}
	}
	return results;
}

export interface Checkpoint {
	messages: Message[];
	state: unknown;
}

// A copy of what a run may change, taken before it starts. A run that does not
// finish is undone to it: question identities are message counts, so the
// messages and the state they advanced are kept or dropped together.
export function checkpoint(messages: Message[], state: unknown): Checkpoint {
	return {
		messages: structuredClone(messages),
		state: structuredClone(state ?? {}),
	};
}

export function textOf(message: Message): string {
	if (message.role !== "user") return "";
	if (typeof message.content === "string") return message.content;
	return message.content
		.map((part) => (part.type === "text" ? part.text : ""))
		.join("");
}

// The run each message belongs to, for the latest turn only: every message
// after the last user message the person sent belongs to `runId`.
export function latestTurnRuns(
	messages: Message[],
	runId: string,
): Record<string, string> {
	const runs: Record<string, string> = {};
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role === "user") break;
		runs[messages[i].id] = runId;
	}
	return runs;
}

// For each run, the index of its last assistant message with text, after
// which that run's evidence renders.
export function evidenceAnchors(
	messages: Message[],
	messageRuns: Record<string, string>,
): Map<number, string> {
	const lastByRun = new Map<string, number>();
	messages.forEach((message, index) => {
		const runId = messageRuns[message.id];
		if (runId && message.role === "assistant" && message.content) {
			lastByRun.set(runId, index);
		}
	});
	return new Map([...lastByRun].map(([runId, index]) => [index, runId]));
}

// What a turn needs from the agent running it.
export interface TurnAgent {
	messages: Message[];
	state: unknown;
	addMessage(message: Message): void;
	setMessages(messages: Message[]): void;
	setState(state: Record<string, unknown>): void;
	runAgent(
		parameters: { runId: string; forwardedProps: Record<string, unknown> },
		subscriber: {
			onRunFinishedEvent: () => void;
			onRunErrorEvent: (params: { event: { message: string } }) => void;
		},
	): Promise<unknown>;
}

export type TurnOutcome =
	| { finished: true }
	| { finished: false; failure: string | null };

// Runs one question. A run is kept only when it finished: a stopped, failed or
// interrupted run is undone to the checkpoint taken before it, since question
// identities are message counts and the messages and the state they advanced
// are kept or dropped together.
export async function runTurn(
	agent: TurnAgent,
	text: string,
	runId: string,
	forwardedProps: Record<string, unknown>,
): Promise<TurnOutcome> {
	const saved = checkpoint(agent.messages, agent.state);
	agent.addMessage({ id: crypto.randomUUID(), role: "user", content: text });
	let finished = false;
	let failure: string | null = null;
	try {
		await agent.runAgent(
			{ runId, forwardedProps },
			{
				onRunFinishedEvent: () => {
					finished = true;
				},
				onRunErrorEvent: ({ event }) => {
					failure = event.message;
				},
			},
		);
	} catch (error) {
		failure ??= error instanceof Error ? error.message : String(error);
	}
	if (finished && failure === null) return { finished: true };
	agent.setMessages(saved.messages);
	agent.setState(saved.state as Record<string, unknown>);
	return { finished: false, failure };
}

const DRAFT_KEY = "haiku.rag.draft.";

export function loadDraft(sessionId: string): string {
	try {
		return sessionStorage.getItem(DRAFT_KEY + sessionId) ?? "";
	} catch {
		return "";
	}
}

export function saveDraft(sessionId: string, text: string): void {
	try {
		if (text) sessionStorage.setItem(DRAFT_KEY + sessionId, text);
		else sessionStorage.removeItem(DRAFT_KEY + sessionId);
	} catch {
		// A browser refusing storage keeps the draft in the page only.
	}
}
