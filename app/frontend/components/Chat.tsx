"use client";

import { CopilotKit, useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import DbInfo from "./DbInfo";

interface ChatSessionState {
	session_id: string;
}

function ChatContent() {
	useCoAgent<ChatSessionState>({
		name: "chat_agent",
		initialState: {
			session_id: "",
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
					width: 100%;
					max-width: 800px;
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
				}
			`}</style>
			<div className="chat-wrapper">
				<div className="chat-container">
					<DbInfo />
					<div className="chat-content">
						<CopilotChat
							labels={{
								title: "haiku.rag Chat",
								initial:
									"Hello! I can help you search and answer questions from your knowledge base. Ask me anything!",
							}}
						/>
					</div>
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
