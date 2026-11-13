import type { ContentPart, Message, ToolCall } from "@ag-ui/client";
import type { DatabaseEntry } from "./databases";
import { newId } from "./id";
import { type Attachment, MAX_IMAGES } from "./images";

export interface Citation {
	index: number;
	document_id: string;
	source?: string | null;
	chunk_id: string;
	chunk_ids?: string[];
	document_uri: string;
	document_title: string | null;
	page_numbers: number[];
	time_range?: [number, number] | null;
	headings: string[] | null;
	content: string;
	doc_item_refs?: string[];
	picture_refs?: string[];
	document_meta?: Record<string, unknown>;
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

// Tool results by the id of the call they answer.
export function toolResults(messages: Message[]): Map<string, string> {
	const results = new Map<string, string>();
	for (const message of messages) {
		if (message.role === "tool") {
			results.set(message.toolCallId, contentText(message.content));
		}
	}
	return results;
}

export type Step =
	| { kind: "reasoning"; id: string; content: string }
	| { kind: "call"; call: ToolCall };

export type Segment =
	| { kind: "message"; message: Message; index: number }
	| { kind: "activity"; key: string; steps: Step[]; messageIds: string[] };

// The conversation as it reads: each unbroken stretch of reasoning and tool
// calls becomes one activity, placed where its first step was.
export function segments(messages: Message[]): Segment[] {
	const out: Segment[] = [];
	let activity: Extract<Segment, { kind: "activity" }> | null = null;
	const step = (id: string, next: Step) => {
		if (!activity) {
			activity = { kind: "activity", key: id, steps: [], messageIds: [] };
			out.push(activity);
		}
		activity.steps.push(next);
		return activity;
	};
	messages.forEach((message, index) => {
		if (message.role === "reasoning") {
			step(message.id, {
				kind: "reasoning",
				id: message.id,
				content: message.content,
			});
		} else if (message.role === "assistant") {
			for (const call of message.toolCalls ?? []) {
				step(call.id, { kind: "call", call }).messageIds.push(message.id);
			}
			if (message.content) {
				activity = null;
				out.push({ kind: "message", message, index });
			}
		} else if (message.role !== "tool") {
			activity = null;
			out.push({ kind: "message", message, index });
		}
	});
	return out;
}

const plural = (count: number, one: string, many: string) =>
	`${count} ${count === 1 ? one : many}`;

// One line for an activity: the searches and programs it ran.
export function activitySummary(steps: Step[]): string {
	const counts = new Map<string, number>();
	for (const step of steps) {
		if (step.kind !== "call" || step.call.function.name === "cite") continue;
		const name = step.call.function.name;
		counts.set(name, (counts.get(name) ?? 0) + 1);
	}
	const parts = [...counts].map(([name, count]) =>
		name === "search"
			? plural(count, "search", "searches")
			: name === "execute_code"
				? plural(count, "program", "programs")
				: `${count} ${name}`,
	);
	return parts.length ? parts.join(", ") : "Reasoning";
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
	return message.role === "user" ? contentText(message.content) : "";
}

function contentText(content: string | ContentPart[]): string {
	if (typeof content === "string") return content;
	return content
		.map((part) => (part.type === "text" ? part.text : ""))
		.join("");
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
	content: string | ContentPart[],
	runId: string,
	forwardedProps: Record<string, unknown>,
): Promise<TurnOutcome> {
	const saved = checkpoint(agent.messages, agent.state);
	agent.addMessage({ id: newId(), role: "user", content });
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

interface Draft {
	text: string;
	images: Attachment[];
}

// What the box holds once a question was not stored: the question, then any
// draft the reader started meanwhile, within the images one question takes.
export function giveBack(box: Draft, question: Draft): Draft {
	return {
		text: [question.text, box.text].filter((text) => text.trim()).join("\n\n"),
		images: [...question.images, ...box.images].slice(0, MAX_IMAGES),
	};
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

// What a question covers: the selected documents, else every available database.
export function scopeLine(
	databases: DatabaseEntry[],
	selected: number,
): string {
	if (selected === 1) return "Ask about the selected document.";
	if (selected > 1) return `Ask about the ${selected} selected documents.`;
	const available = databases.filter((db) => db.available);
	const documents = available.reduce((sum, db) => sum + (db.documents ?? 0), 0);
	const names = new Intl.ListFormat("en").format(
		available.map((db) => db.name),
	);
	const count = `${documents.toLocaleString("en")} ${documents === 1 ? "document" : "documents"}`;
	return `Ask about ${count} in ${names}.`;
}
