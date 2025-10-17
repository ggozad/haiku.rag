"use client";

import {
	CopilotKit,
	useCoAgent,
	useCoAgentStateRender,
} from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import StateDisplay from "./StateDisplay";

interface ResearchState {
	question: string;
	status: string;
	current_iteration: number;
	max_iterations: number;
	confidence: number;
	plan: Array<Record<string, unknown>>;
	findings: Array<Record<string, unknown>>;
	final_report: Record<string, unknown> | null;
}

function AgentContent() {
	// Use useCoAgent to sync state with the backend research agent
	const { state } = useCoAgent<ResearchState>({
		name: "research_agent",
		initialState: {
			question: "",
			status: "idle",
			current_iteration: 0,
			max_iterations: 2,
			confidence: 0.0,
			plan: [],
			findings: [],
			final_report: null,
		},
	});

	// Render state updates from the research agent
	useCoAgentStateRender<ResearchState>({
		name: "research_agent",
		render: ({ state: newState }) => {
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
					<strong>Research Update:</strong> Status: {newState.status},
					Iteration: {newState.current_iteration}/{newState.max_iterations},
					Confidence: {(newState.confidence * 100).toFixed(0)}%
				</div>
			);
		},
	});

	return (
		<div style={{ display: "flex", height: "100vh" }}>
			<div
				style={{
					flex: 1,
					padding: "2rem",
					overflow: "auto",
				}}
			>
				<div
					style={{
						maxWidth: "800px",
						margin: "0 auto",
					}}
				>
					<header style={{ marginBottom: "2rem" }}>
						<h1
							style={{
								fontSize: "2.5rem",
								fontWeight: "bold",
								marginBottom: "0.5rem",
								color: "#1a202c",
							}}
						>
							Haiku.rag Research Assistant
						</h1>
						<p
							style={{
								fontSize: "1.125rem",
								color: "#4a5568",
								lineHeight: "1.6",
							}}
						>
							Interactive research powered by <strong>Haiku.rag</strong>,{" "}
							<strong>Pydantic AI</strong>, and <strong>AG-UI</strong>
						</p>
					</header>

					<StateDisplay state={state} />
				</div>
			</div>

			<CopilotSidebar
				defaultOpen={true}
				clickOutsideToClose={false}
				labels={{
					title: "Research Assistant",
					initial:
						"Hello! I can help you conduct deep research on complex questions using the haiku.rag knowledge base. Ask me anything!",
				}}
			/>
		</div>
	);
}

export default function Agent() {
	return (
		<CopilotKit runtimeUrl="/api/copilotkit" agent="research_agent">
			<AgentContent />
		</CopilotKit>
	);
}
