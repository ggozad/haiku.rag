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

export interface QAResponse {
	question: string;
	answer: string;
	confidence: number;
	citations: Citation[];
}

export interface SessionContext {
	summary: string;
	last_updated: string | null;
}

export interface ChatSessionState {
	initial_context: string | null;
	citations: Citation[];
	qa_history: QAResponse[];
	session_context: SessionContext | null;
	document_filter: string[];
	citation_registry: Record<string, number>;
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
	chatState: ChatSessionState;
	createdAt: string;
	updatedAt: string;
}

const SESSIONS_KEY = "haiku.rag.sessions";
const ACTIVE_SESSION_KEY = "haiku.rag.activeSession";

export function normalizeChatState(state?: ChatSessionState): ChatSessionState {
	return {
		initial_context: state?.initial_context ?? null,
		citations: state?.citations ?? [],
		qa_history: state?.qa_history ?? [],
		session_context: state?.session_context ?? null,
		document_filter: state?.document_filter ?? [],
		citation_registry: state?.citation_registry ?? {},
	};
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
		chatState: normalizeChatState(),
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
	chatState: ChatSessionState,
): void {
	const sessions = getAllSessions();
	const idx = sessions.findIndex((s) => s.id === id);
	if (idx < 0) return;

	const session = sessions[idx];
	session.messages = messages;
	session.chatState = chatState;
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
