import type { Message } from "@ag-ui/client";
import type { DocumentRef, Evidence } from "./chat";

const BASE = "/api/backend/api/sessions";

export interface RunSummary {
	citation_status?: "grounded" | "ungrounded" | "missing";
	violation?: boolean;
}

export interface RunEvidence extends Evidence {
	summary: RunSummary;
}

export interface SessionSummary {
	id: string;
	title: string;
	created_at: string;
	updated_at: string;
}

export interface Session extends SessionSummary {
	messages: Message[];
	state: Record<string, unknown>;
	selection: DocumentRef[];
	runs: Record<string, RunEvidence>;
	message_runs: Record<string, string>;
	inserted: string[];
	in_flight_run_id: string | null;
}

async function json<T>(response: Response): Promise<T> {
	const data = await response.json();
	if (!response.ok) throw new Error(data.error || response.statusText);
	return data as T;
}

export async function listSessions(): Promise<SessionSummary[]> {
	return (await json<{ sessions: SessionSummary[] }>(await fetch(BASE)))
		.sessions;
}

export async function createSession(): Promise<SessionSummary> {
	return json(await fetch(BASE, { method: "POST" }));
}

export async function getSession(id: string): Promise<Session> {
	return json(await fetch(`${BASE}/${encodeURIComponent(id)}`));
}

export async function updateSession(
	id: string,
	change: { title?: string; selection?: DocumentRef[] },
): Promise<Session> {
	return json(
		await fetch(`${BASE}/${encodeURIComponent(id)}`, {
			method: "PATCH",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(change),
		}),
	);
}

export async function deleteSession(id: string): Promise<void> {
	const response = await fetch(`${BASE}/${encodeURIComponent(id)}`, {
		method: "DELETE",
	});
	if (!response.ok) await json(response);
}

export function exportUrl(id: string): string {
	return `${BASE}/${encodeURIComponent(id)}/export`;
}

export type RunOutcome = "stored" | "not-stored" | "pending";

// After a run whose stream was lost, waits until the session no longer holds
// it, then says whether the backend stored it. Membership, not recency: another
// tab may have committed a later turn in between. A run still held when the
// waiting runs out is pending: neither stored nor lost yet.
export async function reconcile(
	id: string,
	runId: string,
	waitMs = 500,
	attempts = 120,
): Promise<{ session: Session; outcome: RunOutcome }> {
	let session = await getSession(id);
	for (let n = 0; session.in_flight_run_id === runId && n < attempts; n++) {
		await new Promise((resolve) => setTimeout(resolve, waitMs));
		session = await getSession(id);
	}
	const outcome: RunOutcome =
		runId in session.runs
			? "stored"
			: session.in_flight_run_id === runId
				? "pending"
				: "not-stored";
	return { session, outcome };
}

const ACTIVE_KEY = "haiku.rag.activeSession";

export function activeSessionId(): string | null {
	try {
		return localStorage.getItem(ACTIVE_KEY);
	} catch {
		return null;
	}
}

export function rememberActiveSession(id: string): void {
	try {
		localStorage.setItem(ACTIVE_KEY, id);
	} catch {
		// A browser refusing storage opens the newest session next time.
	}
}
