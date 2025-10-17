"use client";

import {
	CopilotKit,
	useCoAgent,
	useCoAgentStateRender,
} from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import StateDisplay from "./StateDisplay";

interface ResearchState {
	question: string;
	phase: string; // idle|planning|searching|analyzing|evaluating|done
	status: string;
	plan: Array<{
		id: number;
		question: string;
		status: string; // pending|searching|done
	}>;
	current_question_index: number;
	current_search: {
		query: string;
		type: string;
		results?: Array<{
			chunk: string;
			score: number;
			source: string;
			expanded: boolean;
		}>;
	} | null;
	insights: Array<{
		summary: string;
		confidence: number;
		sources: string[];
	}>;
	confidence: number;
	final_report: {
		title: string;
		summary: string;
		findings: string[];
		conclusions: string[];
		sources: string[];
	} | null;
}

function AgentContent() {
	// Use useCoAgent to sync state with the backend research agent
	const { state } = useCoAgent<ResearchState>({
		name: "research_agent",
		initialState: {
			question: "",
			phase: "idle",
			status: "",
			plan: [],
			current_question_index: 0,
			current_search: null,
			insights: [],
			confidence: 0.0,
			final_report: null,
		},
	});

	// Log state changes
	console.log("[FRONTEND] Current state:", state);

	// Render state updates from the research agent
	useCoAgentStateRender<ResearchState>({
		name: "research_agent",
		render: ({ state: newState }) => {
			console.log("[FRONTEND] State render update:", newState);
			// Show different messages based on phase
			let phaseMessage = "";
			switch (newState.phase) {
				case "planning":
					phaseMessage = "Planning research...";
					break;
				case "searching":
					phaseMessage = newState.current_search
						? `Searching: ${newState.current_search.query}`
						: "Searching...";
					break;
				case "analyzing":
					phaseMessage = "Extracting insights...";
					break;
				case "evaluating":
					phaseMessage = `Evaluating confidence: ${(newState.confidence * 100).toFixed(0)}%`;
					break;
				case "synthesizing":
					phaseMessage = "Generating final report...";
					break;
				case "done":
					phaseMessage = "Research complete!";
					break;
				default:
					phaseMessage = newState.status || "Ready";
			}

			return (
				<div
					style={{
						padding: "1rem",
						background: "#e6f7ff",
						borderRadius: "4px",
						marginBottom: "0.5rem",
						border: "1px solid #91d5ff",
					}}
				>
					<strong>Research Update:</strong> {phaseMessage}
				</div>
			);
		},
	});

	return (
		<>
			<style>{`
				.chat-container {
					width: 50%;
					height: 100vh;
					border-right: 1px solid #e2e8f0;
					display: flex;
					flex-direction: column;
				}
				.chat-container > * {
					flex: 1;
					min-height: 0;
				}
			`}</style>
			<div style={{ display: "flex", height: "100vh" }}>
				{/* Chat on the left */}
				<div className="chat-container">
					<CopilotChat
						labels={{
							title: "Research Assistant",
							initial:
								"Hello! I can help you conduct deep research on complex questions using the haiku.rag knowledge base. Ask me anything!",
						}}
					/>
				</div>

				{/* State display on the right */}
				<div
					style={{
						width: "50%",
						height: "100vh",
						overflow: "auto",
						background: "#f7fafc",
					}}
				>
					<div style={{ padding: "2rem" }}>
						<header style={{ marginBottom: "2rem" }}>
							<h1
								style={{
									fontSize: "2rem",
									fontWeight: "bold",
									marginBottom: "0.5rem",
									color: "#1a202c",
								}}
							>
								Research State
							</h1>
							<p
								style={{
									fontSize: "0.875rem",
									color: "#4a5568",
									lineHeight: "1.6",
								}}
							>
								Live updates from the research agent
							</p>
						</header>

						<StateDisplay state={state} />
					</div>
				</div>
			</div>
		</>
	);
}

export default function Agent() {
	return (
		<CopilotKit runtimeUrl="/api/copilotkit" agent="research_agent">
			<AgentContent />
		</CopilotKit>
	);
}
