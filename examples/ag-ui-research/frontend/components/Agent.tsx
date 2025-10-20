"use client";

import React, { useState } from "react";
import {
	CopilotKit,
	useCoAgent,
	useCoAgentStateRender,
	useCopilotAction,
} from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import StateDisplay from "./StateDisplay";

interface SourceRef {
	chunk_id: string;
	document_uri: string;
	document_title: string;
	chunk_position: number;
}

interface ResearchState {
	question: string;
	phase: string;
	status: string;
	plan: Array<{
		id: number;
		question: string;
		status: string;
		search_results?: {
			type: string;
			results: Array<{
				chunk: string;
				chunk_id: string;
				document_uri: string;
				document_title: string;
				chunk_position: number;
				full_chunk_content: string;
				score: number;
				expanded: boolean;
			}>;
		};
	}>;
	current_question_index: number;
	insights: Array<{
		summary: string;
		confidence: number;
		source_refs: SourceRef[];
	}>;
	document_registry: Record<
		string,
		{
			title: string;
			chunks_referenced: string[];
		}
	>;
	current_document: {
		uri: string;
		title: string;
		content: string;
		total_chunks: number;
		metadata?: Record<string, unknown>;
	} | null;
	confidence: number;
	final_report: {
		title: string;
		summary: string;
		findings: string[];
		conclusions: string[];
		sources: string[];
		citations: Array<{
			document_uri: string;
			document_title: string;
			chunk_ids: string[];
		}>;
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
			insights: [],
			document_registry: {},
			current_document: null,
			confidence: 0.0,
			final_report: null,
		},
	});

	// Log state changes
	console.log("[FRONTEND] Current state:", state);

	// Human-in-the-loop: Request approval for research plan
	console.log("[FRONTEND] Registering approve_research_plan action");
	useCopilotAction({
		name: "approve_research_plan",
		description:
			"Request user approval for the research plan. Returns 'APPROVED' if approved or 'REVISE' if user wants to revise.",
		parameters: [],
		renderAndWaitForResponse: ({ respond, status }) => {
			console.log(
				"[FRONTEND ACTION] renderAndWaitForResponse called",
				{ status }
			);

			return (
				<div
					style={{
						padding: "1.5rem",
						background: "white",
						borderRadius: "8px",
						border: "2px solid #4299e1",
						marginBottom: "1rem",
						boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
					}}
				>
					<h3
						style={{
							fontSize: "1.25rem",
							fontWeight: "bold",
							marginBottom: "1rem",
							color: "#2d3748",
						}}
					>
						Research Plan Approval
					</h3>
					<p
						style={{
							fontSize: "0.875rem",
							color: "#4a5568",
							marginBottom: "1rem",
						}}
					>
						Please review the research plan in the right pane.
					</p>

					<div
						style={{
							display: "flex",
							gap: "1rem",
						}}
						className={status !== "executing" ? "hidden" : ""}
					>
						<button
							type="button"
							onClick={() => respond?.("REVISE")}
							disabled={status !== "executing"}
							style={{
								flex: 1,
								padding: "0.75rem",
								background: "white",
								border: "2px solid #e2e8f0",
								borderRadius: "6px",
								fontSize: "0.875rem",
								fontWeight: "600",
								cursor: status === "executing" ? "pointer" : "not-allowed",
								opacity: status === "executing" ? 1 : 0.5,
							}}
						>
							Revise Plan
						</button>
						<button
							type="button"
							onClick={() => respond?.("APPROVED")}
							disabled={status !== "executing"}
							style={{
								flex: 1,
								padding: "0.75rem",
								background: "#4299e1",
								color: "white",
								border: "none",
								borderRadius: "6px",
								fontSize: "0.875rem",
								fontWeight: "600",
								cursor: status === "executing" ? "pointer" : "not-allowed",
								opacity: status === "executing" ? 1 : 0.5,
							}}
						>
							Approve & Start Research
						</button>
					</div>
				</div>
			);
		},
	});

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
					phaseMessage = "Searching...";
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
