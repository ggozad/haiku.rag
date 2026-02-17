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
import { BrainIcon, FilterIcon } from "../lib/icons";
import type { ChatSessionState } from "../lib/sessionStorage";
import {
	createSession,
	getActiveSessionId,
	getSession,
	normalizeChatState,
	updateSessionMessages,
} from "../lib/sessionStorage";
import CitationBlock from "./CitationBlock";
import ContextPanel from "./ContextPanel";
import DbInfo from "./DbInfo";
import DocumentFilter from "./DocumentFilter";
import SessionManager from "./SessionManager";

// Must match AGUI_STATE_KEY from haiku.rag.agents.chat
const AGUI_STATE_KEY = "haiku.rag.chat";

// AG-UI state is namespaced under AGUI_STATE_KEY
interface AgentState {
	[AGUI_STATE_KEY]?: ChatSessionState;
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
			case "ask":
				return <MessageIcon />;
			case "get_document":
				return <FileIcon />;
			default:
				return <SearchIcon />;
		}
	};

	const getToolLabel = () => {
		switch (toolName) {
			case "search":
				return "Search";
			case "ask":
				return "Ask";
			case "get_document":
				return "Document";
			default:
				return toolName;
		}
	};

	const getDescription = () => {
		switch (toolName) {
			case "search": {
				const query = args.query as string;
				const docName = args.document_name as string | undefined;
				return (
					<>
						<span className="tool-query">{query}</span>
						{docName && (
							<span className="tool-context">
								{" "}
								in <em>{docName}</em>
							</span>
						)}
					</>
				);
			}
			case "ask": {
				const question = args.question as string;
				const docName = args.document_name as string | undefined;
				return (
					<>
						<span className="tool-query">{question}</span>
						{docName && (
							<span className="tool-context">
								{" "}
								from <em>{docName}</em>
							</span>
						)}
					</>
				);
			}
			case "get_document":
				return <span className="tool-query">{args.query as string}</span>;
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

// Context for sharing chat state with the message view
const ChatStateContext = createContext<ChatSessionState | null>(null);

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
	messages,
	isRunning,
}: {
	// biome-ignore lint/suspicious/noExplicitAny: AG-UI Message type is a broad union
	messages: any[];
	isRunning: boolean;
}) {
	const chatState = useContext(ChatStateContext);

	const cursor = isRunning ? (
		<div key="cursor" className="streaming-cursor">
			<span className="dot" />
			<span className="dot" />
			<span className="dot" />
		</div>
	) : null;

	return (
		<CopilotChatMessageView messages={messages} isRunning={isRunning}>
			{({ messageElements }) => {
				if (!chatState?.citations_history?.length) {
					return (
						<>
							{messageElements}
							{cursor}
						</>
					);
				}

				// CopilotChatMessageView renders one element per user/assistant/activity
				// message (tool messages produce nothing). We correlate elements with
				// messages to inject CitationBlocks after the right assistant responses.
				//
				// Both search and ask tools append to citations_history in order,
				// so after each assistant text response that followed tool calls,
				// we inject the next citations_history entry.
				const result: React.ReactNode[] = [];
				let citIdx = 0;
				let seenToolCalls = false;
				let elemIdx = 0;

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

					const isRendered =
						msg.role === "user" ||
						msg.role === "assistant" ||
						msg.role === "activity";
					if (!isRendered) continue;

					if (elemIdx < messageElements.length) {
						result.push(messageElements[elemIdx]);
						elemIdx++;
					}

					// After an assistant text response that followed tool calls,
					// inject the next citations_history entry (one per turn)
					if (msg.role === "assistant" && msg.content && seenToolCalls) {
						if (citIdx < chatState.citations_history.length) {
							const citations = chatState.citations_history[citIdx];
							if (citations?.length) {
								result.push(
									<CitationBlock
										key={`citations-${citIdx}`}
										citations={citations}
									/>,
								);
							}
							citIdx++;
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

function ChatContentInner({
	sessionId,
	onSessionChange,
}: {
	sessionId: string;
	onSessionChange: (id: string) => void;
}) {
	const [contextOpen, setContextOpen] = useState(false);
	const [filterOpen, setFilterOpen] = useState(false);

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

	const chatState = normalizeChatState(
		(agent.state as AgentState)?.[AGUI_STATE_KEY],
	);

	const mergeChatState = (partial: Partial<ChatSessionState>) => {
		const current = normalizeChatState(
			(agent.state as AgentState)?.[AGUI_STATE_KEY],
		);
		agent.setState({
			...agent.state,
			[AGUI_STATE_KEY]: {
				...current,
				...partial,
			},
		});
	};

	// Restore session from localStorage when agent reference changes.
	// useAgent returns a provisional agent initially, then the real agent
	// after runtime connects — re-run restore each time so messages stick.
	useEffect(() => {
		if (agent.messages.length > 0) return;
		const session = getSession(sessionId);
		if (!session) return;
		if (session.chatState) {
			agent.setState({
				[AGUI_STATE_KEY]: normalizeChatState(session.chatState),
			});
		}
		if (session.messages.length > 0) {
			// biome-ignore lint/suspicious/noExplicitAny: AG-UI Message type is a broad union
			agent.setMessages(session.messages as any[]);
		}
	}, [agent, sessionId]);

	// Persist messages and state to localStorage.
	// Read chatState from agent.state at effect time (not render time) so that
	// restore and persist effects in the same commit see consistent state.
	// biome-ignore lint/correctness/useExhaustiveDependencies: JSON.stringify tracks content changes
	useEffect(() => {
		if (sessionId && agent.messages.length > 0) {
			const currentChatState = normalizeChatState(
				(agent.state as AgentState)?.[AGUI_STATE_KEY],
			);
			updateSessionMessages(
				sessionId,
				serializeMessages(agent.messages),
				currentChatState,
			);
		}
	}, [JSON.stringify(agent.messages), chatState, sessionId]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: stable identity via agent ref
	const messages = useMemo(
		() => [...agent.messages],
		[JSON.stringify(agent.messages)],
	);

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

	const sessionContext = chatState.session_context;
	const documentFilter = chatState.document_filter;
	const initialContext = chatState.initial_context ?? "";

	// Context is locked after first message (qa_history has entries)
	const isContextLocked = (chatState.qa_history?.length ?? 0) > 0;

	const handleFilterApply = (selected: string[]) => {
		mergeChatState({ document_filter: selected });
	};

	const handleInitialContextChange = (value: string) => {
		if (isContextLocked) return;
		mergeChatState({ initial_context: value || null });
	};

	return (
		<ChatStateContext.Provider value={chatState}>
			<div className="chat-wrapper">
				<div className="chat-container">
					<div className="chat-header">
						<SessionManager
							activeSessionId={sessionId}
							onSessionChange={onSessionChange}
						/>
						<button
							type="button"
							className={`header-btn ${documentFilter.length > 0 ? "has-content" : ""}`}
							onClick={() => setFilterOpen(true)}
							title={
								documentFilter.length > 0
									? `Filtering: ${documentFilter.length} document(s)`
									: "Filter documents"
							}
						>
							<FilterIcon />
							{documentFilter.length > 0
								? `Filter (${documentFilter.length})`
								: "Filter"}
						</button>
						<button
							type="button"
							className={`header-btn ${initialContext || sessionContext?.summary ? "has-content" : ""}`}
							onClick={() => setContextOpen(true)}
							title={
								isContextLocked
									? sessionContext?.summary
										? "View session context"
										: "No session context yet"
									: initialContext
										? "Edit initial context"
										: "Set initial context"
							}
						>
							<BrainIcon />
							Memory
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
			<ContextPanel
				isOpen={contextOpen}
				onClose={() => setContextOpen(false)}
				sessionContext={sessionContext}
				initialContext={initialContext}
				onInitialContextChange={handleInitialContextChange}
				isLocked={isContextLocked}
			/>
			<DocumentFilter
				isOpen={filterOpen}
				onClose={() => setFilterOpen(false)}
				selected={documentFilter}
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
