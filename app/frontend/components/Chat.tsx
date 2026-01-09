"use client";

import {
	CopilotKit,
	useCoAgent,
	useCoAgentStateRender,
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
