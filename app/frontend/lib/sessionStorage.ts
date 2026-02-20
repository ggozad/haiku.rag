export interface Citation {
	index: number;
	document_id: string;
	chunk_id: string;
	document_uri: string;
	document_title: string | null;
	page_numbers: number[];
	headings: string[] | null;
	content: string;
}

export interface QAHistoryEntry {
	question: string;
	answer: string;
	citations: Citation[];
}

export interface DocumentInfo {
	id: string;
	title: string;
	uri: string;
	created: string;
}

export interface ResearchEntry {
	question: string;
	title: string;
	executive_summary: string;
}

// Matches RAGState from the backend skill
export interface RAGState {
	citations: Citation[];
	qa_history: QAHistoryEntry[];
	document_filter: string | null;
	searches: Record<string, unknown[]>;
	documents: DocumentInfo[];
	reports: ResearchEntry[];
}

export interface StoredMessage {
	id: string;
	role?: string;
	content?: string;
	[key: string]: unknown;
}

export interface StoredSession {
	id: string;
	title: string;
	messages: StoredMessage[];
	ragState: RAGState;
	createdAt: string;
	updatedAt: string;
}

const SESSIONS_KEY = "haiku.rag.sessions";
const ACTIVE_SESSION_KEY = "haiku.rag.activeSession";

export function normalizeRAGState(state?: Partial<RAGState>): RAGState {
	return {
		citations: state?.citations ?? [],
		qa_history: state?.qa_history ?? [],
		document_filter: state?.document_filter ?? null,
		searches: state?.searches ?? {},
		documents: state?.documents ?? [],
		reports: state?.reports ?? [],
	};
}

// Derive per-turn citation arrays from qa_history
export function deriveCitationsHistory(state: RAGState): Citation[][] {
	return state.qa_history
		.filter((entry) => entry.citations?.length > 0)
		.map((entry) => entry.citations);
}

export function getAllSessions(): StoredSession[] {
	const raw = localStorage.getItem(SESSIONS_KEY);
	if (!raw) return [];
	try {
		return JSON.parse(raw) as StoredSession[];
	} catch {
		return [];
	}
}

export function getSession(id: string): StoredSession | null {
	return getAllSessions().find((s) => s.id === id) ?? null;
}

export function getActiveSessionId(): string | null {
	return localStorage.getItem(ACTIVE_SESSION_KEY);
}

export function setActiveSessionId(id: string): void {
	localStorage.setItem(ACTIVE_SESSION_KEY, id);
}

export function createSession(): StoredSession {
	const now = new Date().toISOString();
	const session: StoredSession = {
		id: crypto.randomUUID(),
		title: "New Session",
		messages: [],
		ragState: normalizeRAGState(),
		createdAt: now,
		updatedAt: now,
	};
	const sessions = getAllSessions();
	sessions.unshift(session);
	localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
	setActiveSessionId(session.id);
	return session;
}

export function saveSession(session: StoredSession): void {
	const sessions = getAllSessions();
	const idx = sessions.findIndex((s) => s.id === session.id);
	if (idx >= 0) {
		sessions[idx] = session;
	} else {
		sessions.unshift(session);
	}
	localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}

export function updateSessionMessages(
	id: string,
	messages: StoredMessage[],
	ragState: RAGState,
): void {
	const sessions = getAllSessions();
	const idx = sessions.findIndex((s) => s.id === id);
	if (idx < 0) return;

	const session = sessions[idx];
	session.messages = messages;
	session.ragState = ragState;
	session.updatedAt = new Date().toISOString();

	// Derive title from first user message
	if (session.title === "New Session") {
		const firstUserMsg = messages.find(
			(m) =>
				m.content &&
				typeof m.role === "string" &&
				m.role.toLowerCase() === "user",
		);
		if (firstUserMsg?.content) {
			session.title =
				firstUserMsg.content.length > 60
					? `${firstUserMsg.content.slice(0, 57)}...`
					: firstUserMsg.content;
		}
	}

	sessions[idx] = session;
	localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}

export function deleteSession(id: string): void {
	const sessions = getAllSessions().filter((s) => s.id !== id);
	localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
	if (getActiveSessionId() === id) {
		localStorage.removeItem(ACTIVE_SESSION_KEY);
	}
}

export function exportSessionToMarkdown(session: StoredSession): void {
	const lines: string[] = [`# ${session.title}`, ""];
	for (const msg of session.messages) {
		const role = typeof msg.role === "string" ? msg.role.toLowerCase() : "";
		if (role === "user" && msg.content) {
			lines.push(`**User:** ${msg.content}`, "");
		} else if (role === "assistant" && msg.content) {
			lines.push(`**Assistant:** ${msg.content}`, "");
		}
	}
	const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = `${session.title.replace(/[^a-zA-Z0-9]/g, "_")}.md`;
	a.click();
	URL.revokeObjectURL(url);
}
