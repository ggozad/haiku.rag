"use client";

import { CopilotKit, useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import StateDisplay from "./StateDisplay";

interface InsightRecord {
	id: string;
	summary: string;
	status: string;
	notes?: string;
	supporting_sources: string[];
	originating_questions: string[];
}

interface GapRecord {
	id: string;
	description: string;
	severity: string;
	blocking: boolean;
	resolved: boolean;
	notes?: string;
	supporting_sources: string[];
	resolved_by: string[];
}

interface SearchAnswer {
	query: string;
	answer: string;
	confidence: number;
	context: string[];
	sources: string[];
}

interface ResearchContext {
	original_question: string;
	sub_questions: string[];
	qa_responses: SearchAnswer[];
	insights: InsightRecord[];
	gaps: GapRecord[];
}

interface EvaluationResult {
	confidence: number;
	reasoning: string;
	should_continue: boolean;
	gaps_identified: string[];
	follow_up_questions: string[];
}

interface ResearchReport {
	question: string;
	summary: string;
	findings: string[];
	conclusions: string[];
	insights_used: string[];
	methodology: string;
}

interface ResearchState {
	context: ResearchContext;
	iterations: number;
	max_iterations: number;
	confidence_threshold: number;
	max_concurrency: number;
	last_eval: EvaluationResult | null;
	last_analysis: {
		insights_extracted: InsightRecord[];
		gaps_identified: GapRecord[];
	} | null;
	result?: ResearchReport;
	current_activity?: string;
	current_activity_message?: string;
}

function AgentContent() {
	const { state } = useCoAgent<ResearchState>({
		name: "research_agent",
		initialState: {
			context: {
				original_question: "",
				sub_questions: [],
				qa_responses: [],
				insights: [],
				gaps: [],
			},
			iterations: 0,
			max_iterations: 3,
			confidence_threshold: 0.8,
			max_concurrency: 1,
			last_eval: null,
			last_analysis: null,
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
