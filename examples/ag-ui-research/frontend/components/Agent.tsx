"use client";

import { CopilotKit, useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import DocumentSelector from "./DocumentSelector";
import StateDisplay from "./StateDisplay";

interface Citation {
	document_id: string;
	chunk_id: string;
	document_uri: string;
	document_title?: string;
	page_numbers: number[];
	headings?: string[];
	content: string;
}

interface SearchAnswer {
	query: string;
	answer: string;
	confidence: number;
	cited_chunks: string[];
	citations: Citation[];
}

interface ResearchContext {
	original_question: string;
	sub_questions: string[];
	qa_responses: SearchAnswer[];
}

interface EvaluationResult {
	new_questions: string[];
	confidence_score: number;
	is_sufficient: boolean;
	reasoning: string;
}

interface ResearchReport {
	title: string;
	executive_summary: string;
	main_findings: string[];
	conclusions: string[];
	limitations: string[];
	recommendations: string[];
	sources_summary: string;
}

interface ResearchState {
	context: ResearchContext;
	iterations: number;
	max_iterations: number;
	confidence_threshold: number;
	max_concurrency: number;
	last_eval: EvaluationResult | null;
	result?: ResearchReport;
	current_activity?: string;
	current_activity_message?: string;
	documentFilter?: string[];
}

function AgentContent() {
	const { state, setState, running } = useCoAgent<ResearchState>({
		name: "research_agent",
		initialState: {
			context: {
				original_question: "",
				sub_questions: [],
				qa_responses: [],
			},
			iterations: 0,
			max_iterations: 3,
			confidence_threshold: 0.8,
			max_concurrency: 1,
			last_eval: null,
			documentFilter: [],
		},
	});

	const handleDocumentFilterChange = (ids: string[]) => {
		setState({ ...state, documentFilter: ids });
	};

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

						<div style={{ marginBottom: "1rem" }}>
							<DocumentSelector
								selected={state.documentFilter || []}
								onChange={handleDocumentFilterChange}
								disabled={running}
							/>
						</div>

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
