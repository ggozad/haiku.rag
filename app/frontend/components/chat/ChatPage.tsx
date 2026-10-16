"use client";

import { HttpAgent } from "@ag-ui/client";
import { Filter } from "lucide-react";
import { useEffect, useMemo, useReducer, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
	type DocumentRef,
	evidenceOf,
	latestTurnRuns,
	loadDraft,
	runTurn,
	saveDraft,
} from "@/lib/chat";
import { useDatabases } from "@/lib/databases";
import {
	agentStateOf,
	createSession,
	getActiveSessionId,
	getSession,
	updateSessionMessages,
	updateSessionSelection,
} from "@/lib/sessionStorage";
import Composer from "./Composer";
import FilterDialog from "./FilterDialog";
import MessageList from "./MessageList";
import SessionMenu from "./SessionMenu";

// The browser store keeps one state per session, so the evidence on show is
// the latest turn's, under one run id.
const LATEST_RUN = "latest";
const NO_HIDDEN = new Set<string>();

function ChatSession({
	sessionId,
	onSessionChange,
}: {
	sessionId: string;
	onSessionChange: (id: string) => void;
}) {
	const stored = useMemo(() => getSession(sessionId), [sessionId]);
	const agent = useMemo(
		() =>
			new HttpAgent({
				url: "/api/backend/v1/chat/stream",
				threadId: sessionId,
				// biome-ignore lint/suspicious/noExplicitAny: stored AG-UI messages
				initialMessages: (stored?.messages ?? []) as any[],
				initialState: agentStateOf(stored ?? undefined),
			}),
		[sessionId, stored],
	);
	const { info } = useDatabases();
	const [, rerender] = useReducer((n: number) => n + 1, 0);
	const [running, setRunning] = useState(false);
	const [draft, setDraft] = useState(() => loadDraft(sessionId));
	const [notice, setNotice] = useState<string | null>(null);
	const [selection, setSelection] = useState<DocumentRef[]>(
		stored?.selection ?? [],
	);
	const [filterOpen, setFilterOpen] = useState(false);

	useEffect(() => {
		const { unsubscribe } = agent.subscribe({
			onMessagesChanged: rerender,
			onStateChanged: rerender,
		});
		return () => {
			unsubscribe();
			agent.abortRun();
		};
	}, [agent]);

	const changeDraft = (text: string) => {
		setDraft(text);
		saveDraft(sessionId, text);
	};

	const send = async (text: string) => {
		changeDraft("");
		setNotice(null);
		setRunning(true);
		const outcome = await runTurn(agent, text, crypto.randomUUID(), {
			documents: selection,
		});
		if (outcome.finished) {
			updateSessionMessages(
				sessionId,
				structuredClone(agent.messages),
				structuredClone(agent.state ?? {}),
			);
		} else {
			// Saved as well as set: switching sessions unmounts this one and
			// stops its run, and the question waits in that session's draft.
			saveDraft(sessionId, text);
			setDraft(text);
			setNotice(
				outcome.failure === null
					? "The answer was stopped. Your question is back in the box."
					: `The answer failed: ${outcome.failure}`,
			);
		}
		setRunning(false);
	};

	const applySelection = (next: DocumentRef[]) => {
		setSelection(next);
		updateSessionSelection(sessionId, next);
	};

	return (
		<div className="flex h-full flex-col">
			<div className="flex items-center gap-2 border-b px-4 py-2">
				<SessionMenu sessionId={sessionId} onSessionChange={onSessionChange} />
				<Button
					variant={selection.length ? "secondary" : "ghost"}
					size="sm"
					onClick={() => setFilterOpen(true)}
				>
					<Filter />
					{selection.length
						? `${selection.length} document${selection.length === 1 ? "" : "s"}`
						: "All documents"}
				</Button>
			</div>
			<div className="min-h-0 flex-1 overflow-y-auto">
				<MessageList
					messages={agent.messages}
					running={running}
					evidenceByRun={{ [LATEST_RUN]: evidenceOf(agent.state) }}
					messageRuns={latestTurnRuns(agent.messages, LATEST_RUN)}
					hidden={NO_HIDDEN}
					showSource={(info?.databases.length ?? 0) > 1}
				/>
			</div>
			{notice && (
				<div className="mx-auto w-full max-w-3xl px-4 pb-2">
					<Alert>
						<AlertDescription>{notice}</AlertDescription>
					</Alert>
				</div>
			)}
			<Composer
				draft={draft}
				onDraftChange={changeDraft}
				onSend={send}
				onStop={() => agent.abortRun()}
				running={running}
			/>
			<FilterDialog
				open={filterOpen}
				onOpenChange={setFilterOpen}
				selection={selection}
				onApply={applySelection}
			/>
		</div>
	);
}

export default function ChatPage() {
	const [sessionId, setSessionId] = useState<string | null>(null);

	useEffect(() => {
		const active = getActiveSessionId();
		setSessionId(active && getSession(active) ? active : createSession().id);
	}, []);

	if (!sessionId) return null;
	return (
		<ChatSession
			key={sessionId}
			sessionId={sessionId}
			onSessionChange={setSessionId}
		/>
	);
}
