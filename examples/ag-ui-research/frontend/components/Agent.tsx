"use client";

import { CopilotKit, useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { useState, useEffect } from "react";
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
	awaiting_decision?: boolean;
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
			awaiting_decision: false,
		},
	});

	const [editableQuestions, setEditableQuestions] = useState<string[]>([]);
	const [newQuestion, setNewQuestion] = useState("");
	const [submitting, setSubmitting] = useState(false);

	// Sync editable questions when state changes
	useEffect(() => {
		if (state.awaiting_decision && state.context.sub_questions) {
			setEditableQuestions([...state.context.sub_questions]);
		}
	}, [state.awaiting_decision, state.context.sub_questions]);

	const handleDocumentFilterChange = (ids: string[]) => {
		setState({ ...state, documentFilter: ids });
	};

	const handleRemoveQuestion = (index: number) => {
		setEditableQuestions(editableQuestions.filter((_, i) => i !== index));
	};

	const handleAddQuestion = () => {
		if (newQuestion.trim()) {
			setEditableQuestions([...editableQuestions, newQuestion.trim()]);
			setNewQuestion("");
		}
	};

	const handleDecision = async (action: "search" | "synthesize") => {
		setSubmitting(true);
		try {
			const response = await fetch(
				`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/v1/research/decide`,
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						thread_id: state.context.original_question, // Use as identifier
						action,
						questions: editableQuestions,
					}),
				}
			);
			if (response.ok) {
				setState({ ...state, awaiting_decision: false });
			}
		} catch (error) {
			console.error("Failed to send decision:", error);
		} finally {
			setSubmitting(false);
		}
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

						{/* Document filter - hidden when research is running */}
						{!running && (
							<div style={{ marginBottom: "1rem" }}>
								<DocumentSelector
									selected={state.documentFilter || []}
									onChange={handleDocumentFilterChange}
								/>
							</div>
						)}

						{/* Decision UI when awaiting human input - hidden when report exists, submitting, or not running */}
						{state.awaiting_decision && !submitting && !state.result && running && (
							<div
								style={{
									marginBottom: "1rem",
									background: "#f0f9ff",
									border: "2px solid #0ea5e9",
									borderRadius: "8px",
									padding: "1rem",
								}}
							>
								<div
									style={{
										fontWeight: "bold",
										color: "#0369a1",
										marginBottom: "0.75rem",
										fontSize: "1rem",
									}}
								>
									Research Decision Point
								</div>

								<div style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: "0.75rem" }}>
									{state.context.qa_responses?.length || 0} answers collected | Iteration {state.iterations || 0}
								</div>

								{/* Questions list */}
								<div style={{ marginBottom: "0.75rem" }}>
									<div style={{ fontSize: "0.8rem", color: "#475569", marginBottom: "0.5rem" }}>
										Pending Questions ({editableQuestions.length}):
									</div>
									{editableQuestions.map((q, idx) => (
										<div
											key={idx}
											style={{
												display: "flex",
												alignItems: "center",
												gap: "0.5rem",
												padding: "0.375rem 0.5rem",
												background: "white",
												borderRadius: "4px",
												marginBottom: "0.25rem",
												fontSize: "0.85rem",
											}}
										>
											<span style={{ flex: 1 }}>{q}</span>
											<button
												type="button"
												onClick={() => handleRemoveQuestion(idx)}
												disabled={submitting}
												style={{
													background: "#ef4444",
													color: "white",
													border: "none",
													borderRadius: "4px",
													padding: "0.25rem 0.5rem",
													cursor: submitting ? "not-allowed" : "pointer",
													fontSize: "0.75rem",
												}}
											>
												Remove
											</button>
										</div>
									))}
								</div>

								{/* Add question input */}
								<div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
									<input
										type="text"
										value={newQuestion}
										onChange={(e) => setNewQuestion(e.target.value)}
										placeholder="Add a new question..."
										disabled={submitting}
										style={{
											flex: 1,
											padding: "0.5rem",
											border: "1px solid #cbd5e1",
											borderRadius: "4px",
											fontSize: "0.85rem",
										}}
										onKeyDown={(e) => {
											if (e.key === "Enter") {
												handleAddQuestion();
											}
										}}
									/>
									<button
										type="button"
										onClick={handleAddQuestion}
										disabled={submitting}
										style={{
											background: "#22c55e",
											color: "white",
											border: "none",
											borderRadius: "4px",
											padding: "0.5rem 1rem",
											cursor: submitting ? "not-allowed" : "pointer",
											fontSize: "0.85rem",
										}}
									>
										Add
									</button>
								</div>

								{/* Action buttons */}
								<div style={{ display: "flex", gap: "0.5rem" }}>
									<button
										type="button"
										onClick={() => handleDecision("search")}
										disabled={editableQuestions.length === 0 || submitting}
										style={{
											flex: 1,
											background: editableQuestions.length === 0 || submitting ? "#94a3b8" : "#0ea5e9",
											color: "white",
											border: "none",
											borderRadius: "4px",
											padding: "0.75rem",
											cursor: editableQuestions.length === 0 || submitting ? "not-allowed" : "pointer",
											fontWeight: "bold",
											fontSize: "0.9rem",
										}}
									>
										{submitting ? "Submitting..." : `Search (${editableQuestions.length})`}
									</button>
									<button
										type="button"
										onClick={() => handleDecision("synthesize")}
										disabled={submitting || (state.context.qa_responses?.length || 0) === 0}
										style={{
											flex: 1,
											background: submitting || (state.context.qa_responses?.length || 0) === 0 ? "#94a3b8" : "#8b5cf6",
											color: "white",
											border: "none",
											borderRadius: "4px",
											padding: "0.75rem",
											cursor: submitting || (state.context.qa_responses?.length || 0) === 0 ? "not-allowed" : "pointer",
											fontWeight: "bold",
											fontSize: "0.9rem",
										}}
									>
										Generate Report
									</button>
								</div>
							</div>
						)}

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
