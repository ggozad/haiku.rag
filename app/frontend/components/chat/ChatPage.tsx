"use client";

import { HttpAgent } from "@ag-ui/client";
import { Filter } from "lucide-react";
import { useCallback, useEffect, useReducer, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
	type DocumentRef,
	loadDraft,
	loadPending,
	type PendingRun,
	runTurn,
	saveDraft,
	savePending,
	settlePending,
} from "@/lib/chat";
import { useDatabases } from "@/lib/databases";
import {
	activeSessionId,
	createSession,
	getSession,
	listSessions,
	reconcile,
	rememberActiveSession,
	type Session,
	updateSession,
} from "@/lib/sessions";
import Composer from "./Composer";
import FilterDialog from "./FilterDialog";
import MessageList from "./MessageList";
import SessionMenu from "./SessionMenu";

function ChatSession({
	initial,
	onSessionChange,
}: {
	initial: Session;
	onSessionChange: (id: string) => void;
}) {
	const sessionId = initial.id;
	const [session, setSession] = useState(initial);
	const [agent] = useState(
		() =>
			new HttpAgent({
				url: "/api/backend/v1/chat/stream",
				threadId: sessionId,
				initialMessages: initial.messages,
				initialState: initial.state,
			}),
	);
	const { info } = useDatabases();
	const [, rerender] = useReducer((n: number) => n + 1, 0);
	const [running, setRunning] = useState(false);
	// A run whose outcome is not known yet holds the session: a new run would
	// replace its record, and its question could no longer come back.
	const [unsettled, setUnsettled] = useState(
		() => loadPending(sessionId) !== null,
	);
	const [updating, setUpdating] = useState(false);
	const [draft, setDraft] = useState(() => loadDraft(sessionId));
	const [notice, setNotice] = useState<string | null>(null);
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

	// The stored session is the conversation: after every run the page shows
	// what the backend committed.
	const show = useCallback(
		(stored: Session) => {
			setSession(stored);
			agent.setMessages(stored.messages);
			agent.setState(stored.state);
		},
		[agent],
	);

	const changeDraft = (text: string) => {
		setDraft(text);
		saveDraft(sessionId, text);
	};

	const restore = useCallback(
		(text: string, why: string) => {
			// Saved as well as set: switching sessions unmounts this one and
			// stops its run, and the question waits in that session's draft.
			saveDraft(sessionId, text);
			setDraft(text);
			setNotice(why);
		},
		[sessionId],
	);

	// Settles a run whose stream was lost: the question comes back only when
	// the backend did not store the turn.
	const settle = useCallback(
		async (
			run: PendingRun,
			failure: string | null,
			current: () => boolean = () => true,
		) => {
			try {
				const { session: stored, outcome } = await reconcile(
					sessionId,
					run.runId,
				);
				if (!current()) return;
				show(stored);
				if (outcome === "pending") {
					setNotice(
						"The answer is still being saved. Reload the page in a moment to see it.",
					);
					return;
				}
				settlePending(sessionId, run.runId);
				if (outcome === "stored") {
					setNotice("The connection dropped, but the answer was saved.");
				} else {
					restore(
						run.text,
						failure === null
							? "The answer was stopped. Your question is back in the box."
							: `The answer failed: ${failure}`,
					);
				}
			} catch (error) {
				if (!current()) return;
				setNotice(
					`Whether the answer was saved is not known yet: ${error instanceof Error ? error.message : error}. Reload the page to check.`,
				);
			}
		},
		[sessionId, show, restore],
	);

	// A run a previous page started and never settled. Only the latest effect
	// applies its outcome.
	useEffect(() => {
		const run = loadPending(sessionId);
		if (!run) return;
		let current = true;
		setNotice("Checking whether the last answer was saved.");
		settle(run, null, () => current).finally(() => {
			if (current) setUnsettled(loadPending(sessionId) !== null);
		});
		return () => {
			current = false;
		};
	}, [sessionId, settle]);

	const send = async (text: string) => {
		changeDraft("");
		setNotice(null);
		setRunning(true);
		const run = { runId: crypto.randomUUID(), text };
		savePending(sessionId, run);
		const outcome = await runTurn(agent, text, run.runId, {});
		if (outcome.finished) {
			settlePending(sessionId, run.runId);
			try {
				show(await getSession(sessionId));
			} catch (error) {
				setNotice(
					`The answer was saved, but the session could not be reloaded: ${error instanceof Error ? error.message : error}`,
				);
			}
		} else {
			await settle(run, outcome.failure);
			setUnsettled(loadPending(sessionId) !== null);
		}
		setRunning(false);
	};

	// Sending waits for the update: its response replaces the conversation.
	const applySelection = (selection: DocumentRef[]) => {
		if (updating) return;
		setUpdating(true);
		updateSession(sessionId, { selection })
			.then(show)
			.catch((error: Error) => setNotice(error.message))
			.finally(() => setUpdating(false));
	};

	const evidenceByRun = running ? {} : session.runs;

	return (
		<div className="flex h-full flex-col">
			<div className="flex items-center gap-2 border-b px-4 py-2">
				<SessionMenu
					sessionId={sessionId}
					title={session.title}
					running={running}
					onSessionChange={onSessionChange}
					onRenamed={(title) => setSession((s) => ({ ...s, title }))}
					onError={setNotice}
				/>
				<Button
					variant={session.selection.length ? "secondary" : "ghost"}
					size="sm"
					disabled={running || unsettled || updating}
					onClick={() => setFilterOpen(true)}
				>
					<Filter />
					{session.selection.length
						? `${session.selection.length} document${session.selection.length === 1 ? "" : "s"}`
						: "All documents"}
				</Button>
			</div>
			<div className="min-h-0 flex-1 overflow-y-auto">
				<MessageList
					messages={agent.messages}
					running={running}
					evidenceByRun={evidenceByRun}
					messageRuns={session.message_runs}
					hidden={new Set(session.inserted)}
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
				blocked={unsettled || updating}
			/>
			<FilterDialog
				open={filterOpen}
				onOpenChange={setFilterOpen}
				selection={session.selection}
				onApply={applySelection}
			/>
		</div>
	);
}

export default function ChatPage() {
	const [session, setSession] = useState<Session | null>(null);
	const [error, setError] = useState<string | null>(null);

	const open = useCallback((id: string) => {
		getSession(id)
			.then((stored) => {
				rememberActiveSession(stored.id);
				setSession(stored);
			})
			.catch((err: Error) => setError(err.message));
	}, []);

	useEffect(() => {
		(async () => {
			const sessions = await listSessions();
			const remembered = activeSessionId();
			const known = sessions.find((s) => s.id === remembered) ?? sessions[0];
			open(known ? known.id : (await createSession()).id);
		})().catch((err: Error) => setError(err.message));
	}, [open]);

	if (error) {
		return (
			<div className="mx-auto max-w-3xl p-6">
				<Alert variant="destructive">
					<AlertDescription>
						The sessions could not be loaded: {error}
					</AlertDescription>
				</Alert>
			</div>
		);
	}
	if (!session) return null;
	return (
		<ChatSession key={session.id} initial={session} onSessionChange={open} />
	);
}
