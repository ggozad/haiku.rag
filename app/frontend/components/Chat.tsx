"use client";

import {
	CopilotChatMessageView,
	CopilotChatView,
	CopilotKitProvider,
	defineToolCallRenderer,
	UseAgentUpdate,
	useAgent,
	useCopilotKit,
} from "@copilotkit/react-core/v2";
import {
	createContext,
	useCallback,
	useContext,
	useEffect,
	useMemo,
	useState,
} from "react";
import { FilterIcon } from "../lib/icons";
import type { RAGState } from "../lib/sessionStorage";
import {
	createSession,
	getActiveSessionId,
	getLatestCitations,
	getSession,
	normalizeRAGState,
	updateSessionMessages,
} from "../lib/sessionStorage";
import CitationBlock from "./CitationBlock";
import DbInfo from "./DbInfo";
import DocumentFilter from "./DocumentFilter";
import SessionManager from "./SessionManager";

// Must match state_namespace from haiku.rag.skills.rag
const AGUI_STATE_KEY = "rag";

// AG-UI state is namespaced under AGUI_STATE_KEY
interface AgentState {
	[AGUI_STATE_KEY]?: RAGState;
}

// biome-ignore lint/suspicious/noExplicitAny: CopilotKit message objects vary at runtime
function serializeMessages(messages: any[]): any[] {
	return JSON.parse(JSON.stringify(messages));
}

function SpinnerIcon() {
	return (
		<svg
			width="16"
			height="16"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
			className="tool-spinner"
		>
			<path d="M21 12a9 9 0 1 1-6.219-8.56" />
		</svg>
	);
}

function CheckIcon() {
	return (
		<svg
			width="16"
			height="16"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2.5"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<polyline points="20 6 9 17 4 12" />
		</svg>
	);
}

function SearchIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<circle cx="11" cy="11" r="8" />
			<path d="m21 21-4.3-4.3" />
		</svg>
	);
}

function MessageIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" />
		</svg>
	);
}

function FileIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
			<path d="M14 2v4a2 2 0 0 0 2 2h4" />
		</svg>
	);
}

function ToolCallIndicator({
	toolName,
	status,
	args,
}: {
	toolName: string;
	status: string;
	args: Record<string, unknown>;
}) {
	const isComplete = status === "complete";

	const getToolIcon = () => {
		switch (toolName) {
			case "search":
				return <SearchIcon />;
			case "get_document":
				return <FileIcon />;
			case "execute_skill":
			case "execute_code":
			case "cite":
				return <MessageIcon />;
			default:
				return <SearchIcon />;
		}
	};

	const getToolLabel = () => {
		switch (toolName) {
			case "search":
				return "Search";
			case "get_document":
				return "Document";
			case "execute_skill":
				return "Skill";
			case "execute_code":
				return "Code";
			case "cite":
				return "Cite";
			case "list_documents":
				return "Documents";
			default:
				return toolName;
		}
	};

	const getDescription = () => {
		switch (toolName) {
			case "execute_skill": {
				const skill = args.skill_name as string | undefined;
				const request = args.request as string | undefined;
				return (
					<span className="tool-query">
						{skill ? `${skill}: ` : ""}
						{request ?? "Processing..."}
					</span>
				);
			}
			case "search": {
				const query = args.query as string;
				return <span className="tool-query">{query}</span>;
			}
			case "get_document":
				return <span className="tool-query">{args.query as string}</span>;
			case "execute_code": {
				const code = args.code as string | undefined;
				return (
					<span className="tool-query">
						{code ? code.slice(0, 80) : "Running code..."}
					</span>
				);
			}
			case "cite":
				return <span className="tool-query">Registering citations</span>;
			case "list_documents":
				return <span className="tool-query">Listing documents</span>;
			default:
				return <span>Processing...</span>;
		}
	};

	return (
		<div className={`tool-call-card ${isComplete ? "complete" : "loading"}`}>
			<div className="tool-status-icon">
				{isComplete ? <CheckIcon /> : <SpinnerIcon />}
			</div>
			<div className="tool-content">
				<div className="tool-header">
					<span className="tool-badge">
						{getToolIcon()}
						{getToolLabel()}
					</span>
					<span className="tool-status-text">
						{isComplete ? "Done" : "Working..."}
					</span>
				</div>
				<div className="tool-description">{getDescription()}</div>
			</div>
		</div>
	);
}

// Render an activity message from a skill sub-agent tool call/result
function ActivityIndicator({
	message,
	isComplete,
}: {
	// biome-ignore lint/suspicious/noExplicitAny: AG-UI activity message shape
	message: any;
	isComplete: boolean;
}) {
	const content = message.content ?? {};
	const toolName = content.tool_name ?? "tool";

	let args: Record<string, unknown> = {};
	if (content.args) {
		try {
			args =
				typeof content.args === "string"
					? JSON.parse(content.args)
					: content.args;
		} catch {
			// ignore parse errors
		}
	}

	return (
		<ToolCallIndicator
			toolName={toolName}
			status={isComplete ? "complete" : "loading"}
			args={args}
		/>
	);
}

// Context for sharing chat state with the message view
const ChatStateContext = createContext<RAGState | null>(null);

// Wildcard tool call renderer for all server-side tools
const toolCallRenderers = [
	defineToolCallRenderer({
		name: "*",
		render: ({ name, args, result }) => (
			<ToolCallIndicator
				toolName={name}
				status={result !== undefined ? "complete" : "loading"}
				args={(args ?? {}) as Record<string, unknown>}
			/>
		),
	}),
];

// Custom message view that injects CitationBlocks after assistant responses.
// Uses CopilotChatMessageView's children render prop to post-process the
// rendered message elements and inject citations at the right positions.
function MessageViewWithCitations({
	messages = [],
	isRunning = false,
}: {
	// biome-ignore lint/suspicious/noExplicitAny: AG-UI Message type is a broad union
	messages?: any[];
	isRunning?: boolean;
}) {
	const ragState = useContext(ChatStateContext);
	const latestCitations = ragState ? getLatestCitations(ragState) : [];

	// Collect completed tool_call_ids from skill_tool_result activity messages
	const completedToolCallIds = useMemo(() => {
		const ids = new Set<string>();
		for (const msg of messages) {
			if (
				msg.role === "activity" &&
				msg.activityType === "skill_tool_result" &&
				msg.content?.tool_call_id
			) {
				ids.add(msg.content.tool_call_id);
			}
		}
		return ids;
	}, [messages]);

	const cursor = isRunning ? (
		<div key="cursor" className="streaming-cursor">
			<span className="dot" />
			<span className="dot" />
			<span className="dot" />
		</div>
	) : null;

	// CopilotChatMessageView renders one element per user/assistant message.
	// We interleave activity indicators (skill sub-agent tool calls) and
	// optionally inject CitationBlocks after assistant responses that
	// followed tool calls.
	return (
		<CopilotChatMessageView messages={messages} isRunning={isRunning}>
			{({ messageElements }) => {
				const result: React.ReactNode[] = [];
				let elemIdx = 0;
				let seenToolCalls = false;

				for (const msg of messages) {
					if (msg.role === "user") {
						seenToolCalls = false;
					}

					if (
						msg.role === "assistant" &&
						Array.isArray(msg.toolCalls) &&
						msg.toolCalls.length > 0
					) {
						seenToolCalls = true;
					}

					// Activity messages are not rendered by CopilotKit —
					// render them ourselves without consuming messageElements
					if (msg.role === "activity") {
						if (msg.activityType === "skill_tool_call") {
							const toolCallId = msg.content?.tool_call_id;
							result.push(
								<ActivityIndicator
									key={`activity-${msg.id}`}
									message={msg}
									isComplete={
										toolCallId ? completedToolCallIds.has(toolCallId) : false
									}
								/>,
							);
						}
						continue;
					}

					if (msg.role !== "user" && msg.role !== "assistant") continue;

					if (elemIdx < messageElements.length) {
						result.push(messageElements[elemIdx]);
						elemIdx++;
					}

					// After an assistant text response that followed tool calls,
					// show citations from the latest turn
					if (msg.role === "assistant" && msg.content && seenToolCalls) {
						if (latestCitations.length > 0) {
							result.push(
								<CitationBlock
									key={`citations-${msg.id}`}
									citations={latestCitations}
								/>,
							);
						}
						seenToolCalls = false;
					}
				}

				while (elemIdx < messageElements.length) {
					result.push(messageElements[elemIdx]);
					elemIdx++;
				}

				return (
					<>
						{result}
						{cursor}
					</>
				);
			}}
		</CopilotChatMessageView>
	);
}
MessageViewWithCitations.Cursor = CopilotChatMessageView.Cursor;

function ChatContentInner({
	sessionId,
	onSessionChange,
}: {
	sessionId: string;
	onSessionChange: (id: string) => void;
}) {
	const [filterOpen, setFilterOpen] = useState(false);
	// Track selected document names locally (frontend-only)
	const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);

	const { agent } = useAgent({
		agentId: "chat_agent",
		updates: [
			UseAgentUpdate.OnMessagesChanged,
			UseAgentUpdate.OnStateChanged,
			UseAgentUpdate.OnRunStatusChanged,
		],
	});
	const { copilotkit: ck } = useCopilotKit();

	// Set threadId (CopilotChat normally does this in its connect effect)
	useEffect(() => {
		agent.threadId = sessionId;
	}, [agent, sessionId]);

	const ragState = normalizeRAGState(
		(agent.state as AgentState)?.[AGUI_STATE_KEY],
	);

	// Restore session from localStorage when agent reference changes.
	// useAgent returns a provisional agent initially, then the real agent
	// after runtime connects — re-run restore each time so messages stick.
	useEffect(() => {
		if (agent.messages.length > 0) return;
		const session = getSession(sessionId);
		if (!session) return;
		if (session.ragState) {
			agent.setState({
				[AGUI_STATE_KEY]: normalizeRAGState(session.ragState),
			});
		}
		if (session.messages.length > 0) {
			// biome-ignore lint/suspicious/noExplicitAny: AG-UI Message type is a broad union
			agent.setMessages(session.messages as any[]);
		}
	}, [agent, sessionId]);

	// Persist messages and state to localStorage.
	// Read ragState from agent.state at effect time (not render time) so that
	// restore and persist effects in the same commit see consistent state.
	// biome-ignore lint/correctness/useExhaustiveDependencies: JSON.stringify tracks content changes
	useEffect(() => {
		if (sessionId && agent.messages.length > 0) {
			const currentRagState = normalizeRAGState(
				(agent.state as AgentState)?.[AGUI_STATE_KEY],
			);
			updateSessionMessages(
				sessionId,
				serializeMessages(agent.messages),
				currentRagState,
			);
		}
	}, [JSON.stringify(agent.messages), ragState, sessionId]);

	// Deduplicate messages by id, keeping the last occurrence.
	// haiku.skills 0.10.0+ sends activity snapshots with the same id
	// and replace=true — CopilotKit doesn't deduplicate these, so we
	// must do it to avoid React duplicate key warnings.
	// biome-ignore lint/correctness/useExhaustiveDependencies: stable identity via agent ref
	const messages = useMemo(() => {
		const seen = new Map<string, number>();
		const msgs = agent.messages;
		for (let i = 0; i < msgs.length; i++) {
			const id = msgs[i].id;
			if (id) seen.set(id, i);
		}
		return msgs.filter((msg, i) => !msg.id || seen.get(msg.id) === i);
	}, [JSON.stringify(agent.messages)]);

	const onSubmitMessage = useCallback(
		async (text: string) => {
			agent.addMessage({
				id: crypto.randomUUID(),
				role: "user",
				content: text,
			});
			try {
				await ck.runAgent({ agent });
			} catch (error) {
				console.error("runAgent failed", error);
			}
		},
		[agent, ck],
	);

	const onStop = useCallback(() => {
		try {
			ck.stopAgent({ agent });
		} catch {
			agent.abortRun();
		}
	}, [agent, ck]);

	const handleFilterApply = (selected: string[]) => {
		setSelectedDocuments(selected);
		// Convert selected document names to SQL filter for the backend
		const filter =
			selected.length > 0
				? selected
						.map(
							(name) =>
								`(title LIKE '%${name.replace(/'/g, "''")}%' OR uri LIKE '%${name.replace(/'/g, "''")}%')`,
						)
						.join(" OR ")
				: null;
		agent.setState({
			...agent.state,
			[AGUI_STATE_KEY]: {
				...ragState,
				document_filter: filter,
			},
		});
	};

	return (
		<ChatStateContext.Provider value={ragState}>
			<div className="chat-wrapper">
				<div className="chat-container">
					<div className="chat-header">
						<SessionManager
							activeSessionId={sessionId}
							onSessionChange={onSessionChange}
						/>
						<button
							type="button"
							className={`header-btn ${selectedDocuments.length > 0 ? "has-content" : ""}`}
							onClick={() => setFilterOpen(true)}
							title={
								selectedDocuments.length > 0
									? `Filtering: ${selectedDocuments.length} document(s)`
									: "Filter documents"
							}
						>
							<FilterIcon />
							{selectedDocuments.length > 0
								? `Filter (${selectedDocuments.length})`
								: "Filter"}
						</button>
					</div>
					<div className="chat-content">
						<CopilotChatView
							messageView={MessageViewWithCitations}
							messages={messages}
							isRunning={agent.isRunning}
							onSubmitMessage={onSubmitMessage}
							onStop={onStop}
						>
							{({ scrollView, input }) => (
								<div className="chat-layout">
									<div className="chat-scroll-area">{scrollView}</div>
									<div className="chat-input-area">{input}</div>
								</div>
							)}
						</CopilotChatView>
					</div>
					<DbInfo />
				</div>
			</div>
			<DocumentFilter
				isOpen={filterOpen}
				onClose={() => setFilterOpen(false)}
				selected={selectedDocuments}
				onApply={handleFilterApply}
			/>
		</ChatStateContext.Provider>
	);
}

export default function Chat() {
	const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

	useEffect(() => {
		let id = getActiveSessionId();
		if (!id) {
			id = createSession().id;
		}
		setActiveSessionId(id);
	}, []);

	if (!activeSessionId) return null;

	return (
		<CopilotKitProvider
			key={activeSessionId}
			runtimeUrl="/api/copilotkit"
			useSingleEndpoint
			renderToolCalls={toolCallRenderers}
		>
			<ChatContentInner
				sessionId={activeSessionId}
				onSessionChange={setActiveSessionId}
			/>
		</CopilotKitProvider>
	);
}
