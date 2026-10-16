import type { InputContent, Message } from "@ag-ui/client";
import type { Attachment } from "./images";

export interface Citation {
	index: number;
	document_id: string;
	source?: string | null;
	chunk_id: string;
	chunk_ids?: string[];
	document_uri: string;
	document_title: string | null;
	page_numbers: number[];
	headings: string[] | null;
	content: string;
	doc_item_refs?: string[];
	picture_refs?: string[];
}

// The fields of the capability's state this UI reads. The capability owns the
// rest of its namespace, and the backend keeps the whole state.
interface RAGState {
	citation_index: Record<string, Citation>;
	citations: string[];
	executions?: unknown[];
}

// A selected document: ids repeat only between copies of a database, so the
// database name is part of its identity.
export interface DocumentRef {
	source: string;
	id: string;
}

export interface RunSummary {
	citation_status?: "grounded" | "ungrounded" | "missing";
	violation?: boolean;
}

export interface Evidence {
	citations: Citation[];
	// The last program the model ran successfully for the question, if any.
	program: string | null;
	summary?: RunSummary;
}

interface CodeExecution {
	code: string;
	success?: boolean;
}

export function ragStateOf(state: unknown): RAGState {
	const rag = (state as Record<string, unknown> | undefined)?.rag as
		| Partial<RAGState>
		| undefined;
	return {
		citation_index: rag?.citation_index ?? {},
		citations: rag?.citations ?? [],
		executions: rag?.executions ?? [],
	};
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
	content: string | InputContent[],
	runId: string,
	forwardedProps: Record<string, unknown>,
): Promise<TurnOutcome> {
	const saved = checkpoint(agent.messages, agent.state);
	agent.addMessage({ id: crypto.randomUUID(), role: "user", content });
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

const DRAFT_IMAGES_KEY = "haiku.rag.draft-images.";

export function loadDraftImages(sessionId: string): Attachment[] {
	try {
		const raw = sessionStorage.getItem(DRAFT_IMAGES_KEY + sessionId);
		return raw ? (JSON.parse(raw) as Attachment[]) : [];
	} catch {
		return [];
	}
}

// False when the browser refused them: they live in the page only.
export function saveDraftImages(
	sessionId: string,
	images: Attachment[],
): boolean {
	try {
		if (images.length)
			sessionStorage.setItem(
				DRAFT_IMAGES_KEY + sessionId,
				JSON.stringify(images),
			);
		else sessionStorage.removeItem(DRAFT_IMAGES_KEY + sessionId);
		return true;
	} catch {
		return false;
	}
}

const PENDING_KEY = "haiku.rag.pending.";

// The run a page started and has not settled, kept so a reload can still ask
// the backend whether it was stored and give the question back.
export interface PendingRun {
	runId: string;
	text: string;
	images?: Attachment[];
	// Storage refused the images, so a reload gives back the text alone.
	imagesDropped?: boolean;
}

export function savePending(sessionId: string, run: PendingRun): void {
	const key = PENDING_KEY + sessionId;
	try {
		sessionStorage.setItem(key, JSON.stringify(run));
	} catch {
		try {
			sessionStorage.setItem(
				key,
				JSON.stringify({ ...run, images: [], imagesDropped: true }),
			);
		} catch {
			// A browser refusing storage settles the run in this page only.
		}
	}
}

// Forgets the run only while it is still the session's pending one.
export function settlePending(sessionId: string, runId: string): void {
	try {
		if (loadPending(sessionId)?.runId === runId)
			sessionStorage.removeItem(PENDING_KEY + sessionId);
	} catch {
		// A browser refusing storage settles the run in this page only.
	}
}

export function loadPending(sessionId: string): PendingRun | null {
	try {
		const raw = sessionStorage.getItem(PENDING_KEY + sessionId);
		return raw ? (JSON.parse(raw) as PendingRun) : null;
	} catch {
		return null;
	}
}

export interface Status {
	kind: "breach" | "grounded" | "ungrounded";
	label: string;
	variant: "secondary" | "outline" | "destructive";
}

// What an answer declared as its grounding, in words. None for an answer that
// declared nothing without breaching the policy: a greeting, an aside.
export function statusOf(summary: RunSummary): Status | null {
	if (summary.violation) {
		return {
			kind: "breach",
			label: "Answered without declaring its sources",
			variant: "destructive",
		};
	}
	if (summary.citation_status === "grounded") {
		return {
			kind: "grounded",
			label: "Grounded in the sources",
			variant: "secondary",
		};
	}
	if (summary.citation_status === "ungrounded") {
		return {
			kind: "ungrounded",
			label: "Declared no sources",
			variant: "outline",
		};
	}
	return null;
}
