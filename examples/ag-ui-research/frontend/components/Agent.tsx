"use client";

import {
	CopilotKit,
	useCoAgent,
	useCopilotAction,
	useCopilotContext,
} from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { useState } from "react";
import DocumentSelector from "./DocumentSelector";
import StateDisplay from "./StateDisplay";

interface SearchAnswer {
	query: string;
	answer: string;
	confidence: number;
	cited_chunks: string[];
	citations: {
		document_id: string;
		chunk_id: string;
		document_uri: string;
		document_title?: string;
		page_numbers: number[];
		headings?: string[];
		content: string;
	}[];
}

interface ResearchState {
	context: {
		original_question: string;
		sub_questions: string[];
		qa_responses: SearchAnswer[];
	};
	iterations: number;
	max_iterations: number;
	confidence_threshold: number;
	max_concurrency: number;
	last_eval: {
		new_questions: string[];
		confidence_score: number;
		is_sufficient: boolean;
		reasoning: string;
	} | null;
	result?: {
		title: string;
		executive_summary: string;
		main_findings: string[];
		conclusions: string[];
		limitations: string[];
		recommendations: string[];
		sources_summary: string;
	};
	current_activity?: string;
	current_activity_message?: string;
	documentFilter?: string[];
}

interface DecisionArgs {
	original_question: string;
	sub_questions: string[];
	qa_responses: SearchAnswer[];
}

type DecisionAction = "search" | "synthesize" | "modify_questions";

interface DecisionResult {
	action: DecisionAction;
	questions?: string[];
}

function DecisionUI({
	args,
	onResolve,
}: {
	args: DecisionArgs;
	onResolve: (result: DecisionResult) => void | Promise<void>;
}) {
	const [editableQuestions, setEditableQuestions] = useState<string[]>(
		args.sub_questions || [],
	);
	const [newQuestion, setNewQuestion] = useState("");
	const [submitting, setSubmitting] = useState(false);

	const qaCount = args.qa_responses?.length || 0;
	const hasQuestions = editableQuestions.length > 0;
	const canSearch = hasQuestions && !submitting;
	const canSynthesize = qaCount > 0 && !submitting;

	const questionsModified =
		editableQuestions.length !== args.sub_questions.length ||
		editableQuestions.some((q, i) => q !== args.sub_questions[i]);

	const handleSubmit = (action: DecisionAction, questions?: string[]) => {
		setSubmitting(true);
		onResolve({ action, questions });
	};

	const handleSearch = () => {
		handleSubmit(
			questionsModified ? "modify_questions" : "search",
			editableQuestions,
		);
	};

	const handleSynthesize = () => {
		handleSubmit("synthesize");
	};

	const handleRemoveQuestion = (index: number) => {
		if (submitting) return;
		setEditableQuestions(editableQuestions.filter((_, i) => i !== index));
	};

	const handleAddQuestion = () => {
		if (submitting || !newQuestion.trim()) return;
		setEditableQuestions([...editableQuestions, newQuestion.trim()]);
		setNewQuestion("");
	};

	if (submitting) {
		return null;
	}

	return (
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

			<div
				style={{
					fontSize: "0.85rem",
					color: "#64748b",
					marginBottom: "0.75rem",
				}}
			>
				{qaCount} answers collected
			</div>

			<div style={{ marginBottom: "0.75rem" }}>
				<div
					style={{
						fontSize: "0.8rem",
						color: "#475569",
						marginBottom: "0.5rem",
					}}
				>
					Pending Questions ({editableQuestions.length}):
				</div>
				{editableQuestions.map((q, idx) => (
					<div
						key={`question-${idx}`}
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
							style={{
								background: "#ef4444",
								color: "white",
								border: "none",
								borderRadius: "4px",
								padding: "0.25rem 0.5rem",
								cursor: "pointer",
								fontSize: "0.75rem",
							}}
						>
							Remove
						</button>
					</div>
				))}
			</div>

			<div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
				<input
					type="text"
					value={newQuestion}
					onChange={(e) => setNewQuestion(e.target.value)}
					placeholder="Add a new question..."
					style={{
						flex: 1,
						padding: "0.5rem",
						border: "1px solid #cbd5e1",
						borderRadius: "4px",
						fontSize: "0.85rem",
					}}
					onKeyDown={(e) => {
						if (e.key === "Enter") handleAddQuestion();
					}}
				/>
				<button
					type="button"
					onClick={handleAddQuestion}
					style={{
						background: "#22c55e",
						color: "white",
						border: "none",
						borderRadius: "4px",
						padding: "0.5rem 1rem",
						cursor: "pointer",
						fontSize: "0.85rem",
					}}
				>
					Add
				</button>
			</div>

			<div style={{ display: "flex", gap: "0.5rem" }}>
				<button
					type="button"
					onClick={handleSearch}
					disabled={!canSearch}
					style={{
						flex: 1,
						background: canSearch ? "#0ea5e9" : "#94a3b8",
						color: "white",
						border: "none",
						borderRadius: "4px",
						padding: "0.75rem",
						cursor: canSearch ? "pointer" : "not-allowed",
						fontWeight: "bold",
						fontSize: "0.9rem",
					}}
				>
					Search ({editableQuestions.length})
				</button>
				<button
					type="button"
					onClick={handleSynthesize}
					disabled={!canSynthesize}
					style={{
						flex: 1,
						background: canSynthesize ? "#8b5cf6" : "#94a3b8",
						color: "white",
						border: "none",
						borderRadius: "4px",
						padding: "0.75rem",
						cursor: canSynthesize ? "pointer" : "not-allowed",
						fontWeight: "bold",
						fontSize: "0.9rem",
					}}
				>
					Generate Report
				</button>
			</div>
		</div>
	);
}

const BACKEND_URL =
	process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

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

	const { threadId } = useCopilotContext();

	const handleDocumentFilterChange = (ids: string[]) => {
		setState({ ...state, documentFilter: ids });
	};

	const sendToolResult = async (result: DecisionResult) => {
		if (!threadId) {
			console.error("No threadId available to send tool result");
			return;
		}

		try {
			const response = await fetch(`${BACKEND_URL}/v1/research/stream`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					threadId,
					messages: [
						{
							id: crypto.randomUUID(),
							role: "tool",
							content: JSON.stringify(result),
						},
					],
				}),
			});

			if (!response.ok) {
				console.error("Failed to send tool result:", response.status);
			}
		} catch (error) {
			console.error("Error sending tool result:", error);
		}
	};

	useCopilotAction({
		name: "human_decision",
		description: "Pause for human decision on research direction",
		parameters: [
			{
				name: "original_question",
				type: "string",
				description: "The original research question",
			},
			{
				name: "sub_questions",
				type: "string[]",
				description: "Pending sub-questions to search",
			},
			{
				name: "qa_responses",
				type: "object[]",
				description: "Answers collected so far",
			},
		],
		renderAndWaitForResponse: ({ args, status }) => {
			if (status === "complete") {
				return null;
			}

			return (
				<DecisionUI
					args={args as unknown as DecisionArgs}
					onResolve={sendToolResult}
				/>
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
				<div className="chat-container">
					<CopilotChat
						labels={{
							title: "Research Assistant",
							initial:
								"Hello! I can help you conduct deep research on complex questions using the haiku.rag knowledge base. Ask me anything!",
						}}
					/>
				</div>

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

						{!running && (
							<div style={{ marginBottom: "1rem" }}>
								<DocumentSelector
									selected={state.documentFilter || []}
									onChange={handleDocumentFilterChange}
								/>
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
