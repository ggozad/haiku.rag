"use client";

import {
	CopilotKit,
	useCoAgent,
	useCoAgentStateRender,
	useCopilotAction,
} from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import CitationBlock from "./CitationBlock";
import DbInfo from "./DbInfo";

interface Citation {
	index: number;
	document_id: string;
	chunk_id: string;
	document_uri: string;
	document_title: string | null;
	page_numbers: number[];
	headings: string[] | null;
	content: string;
}

interface QAResponse {
	question: string;
	answer: string;
	confidence: number;
	citations: Citation[];
}

interface ChatSessionState {
	session_id: string;
	citations: Citation[];
	qa_history: QAResponse[];
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
			<style>{`
				@keyframes spin {
					from { transform: rotate(0deg); }
					to { transform: rotate(360deg); }
				}
				@keyframes fadeIn {
					from { opacity: 0; transform: translateY(-4px); }
					to { opacity: 1; transform: translateY(0); }
				}
				@keyframes pulse {
					0%, 100% { opacity: 1; }
					50% { opacity: 0.6; }
				}
				.tool-call-card {
					display: flex;
					align-items: flex-start;
					gap: 12px;
					padding: 12px 14px;
					margin: 8px 0;
					background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
					border-radius: 10px;
					font-size: 13px;
					color: #475569;
					border: 1px solid #e2e8f0;
					box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
					animation: fadeIn 0.2s ease-out;
					transition: all 0.2s ease;
				}
				.tool-call-card.loading {
					border-color: #bfdbfe;
					background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
				}
				.tool-call-card.complete {
					border-color: #bbf7d0;
					background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
				}
				.tool-status-icon {
					display: flex;
					align-items: center;
					justify-content: center;
					width: 28px;
					height: 28px;
					border-radius: 8px;
					flex-shrink: 0;
				}
				.tool-call-card.loading .tool-status-icon {
					background: #dbeafe;
					color: #2563eb;
				}
				.tool-call-card.complete .tool-status-icon {
					background: #bbf7d0;
					color: #16a34a;
				}
				.tool-spinner {
					animation: spin 1s linear infinite;
				}
				.tool-content {
					flex: 1;
					min-width: 0;
				}
				.tool-header {
					display: flex;
					align-items: center;
					gap: 6px;
					margin-bottom: 4px;
				}
				.tool-badge {
					display: inline-flex;
					align-items: center;
					gap: 4px;
					padding: 2px 8px;
					background: rgba(59, 130, 246, 0.1);
					color: #2563eb;
					border-radius: 4px;
					font-size: 11px;
					font-weight: 600;
					text-transform: uppercase;
					letter-spacing: 0.025em;
				}
				.tool-call-card.complete .tool-badge {
					background: rgba(22, 163, 74, 0.1);
					color: #16a34a;
				}
				.tool-status-text {
					font-size: 11px;
					color: #94a3b8;
				}
				.tool-call-card.loading .tool-status-text {
					animation: pulse 1.5s ease-in-out infinite;
				}
				.tool-description {
					color: #334155;
					line-height: 1.5;
					word-break: break-word;
				}
				.tool-query {
					color: #0f172a;
					font-weight: 500;
				}
				.tool-context {
					color: #64748b;
				}
				.tool-context em {
					color: #475569;
					font-style: normal;
					font-weight: 500;
				}
			`}</style>
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

function ChatContent() {
	useCoAgent<ChatSessionState>({
		name: "chat_agent",
		initialState: {
			session_id: "",
			citations: [],
			qa_history: [],
		},
	});

	useCoAgentStateRender<ChatSessionState>({
		name: "chat_agent",
		render: ({ state }) => {
			if (state.citations && state.citations.length > 0) {
				return <CitationBlock citations={state.citations} />;
			}
			return null;
		},
	});

	useCopilotAction({
		name: "search",
		available: "disabled",
		parameters: [
			{ name: "query", type: "string" },
			{ name: "document_name", type: "string" },
		],
		render: ({ status, args }) => (
			<ToolCallIndicator
				toolName="search"
				status={status}
				args={args as Record<string, unknown>}
			/>
		),
	});

	useCopilotAction({
		name: "ask",
		available: "disabled",
		parameters: [
			{ name: "question", type: "string" },
			{ name: "document_name", type: "string" },
		],
		render: ({ status, args }) => (
			<ToolCallIndicator
				toolName="ask"
				status={status}
				args={args as Record<string, unknown>}
			/>
		),
	});

	useCopilotAction({
		name: "get_document",
		available: "disabled",
		parameters: [{ name: "query", type: "string" }],
		render: ({ status, args }) => (
			<ToolCallIndicator
				toolName="get_document"
				status={status}
				args={args as Record<string, unknown>}
			/>
		),
	});

	return (
		<>
			<style>{`
				.chat-wrapper {
					display: flex;
					justify-content: center;
					align-items: center;
					min-height: 100vh;
					padding: 1rem;
				}
				.chat-container {
					width: calc(100% - 2rem);
					max-width: 1400px;
					height: 90vh;
					border-radius: 12px;
					overflow: hidden;
					box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
					background: white;
					display: flex;
					flex-direction: column;
				}
				.chat-content {
					flex: 1;
					min-height: 0;
					display: flex;
					flex-direction: column;
				}
				.chat-content > * {
					flex: 1;
					min-height: 0;
				}
			`}</style>
			<div className="chat-wrapper">
				<div className="chat-container">
					<div className="chat-content">
						<CopilotChat
							labels={{
								title: "haiku.rag Chat",
								initial:
									"Hello! I can help you search and answer questions from your knowledge base. Ask me anything!",
							}}
						/>
					</div>
					<DbInfo />
				</div>
			</div>
		</>
	);
}

export default function Chat() {
	return (
		<CopilotKit runtimeUrl="/api/copilotkit" agent="chat_agent">
			<ChatContent />
		</CopilotKit>
	);
}
