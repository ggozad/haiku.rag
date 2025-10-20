"use client";

import { Markdown } from "@copilotkit/react-ui";
import { useState } from "react";

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

interface StateDisplayProps {
	state: ResearchState;
}

export default function StateDisplay({ state }: StateDisplayProps) {
	const [expandedSections, setExpandedSections] = useState<
		Record<string, boolean>
	>({
		plan: true,
		insights: true,
		report: true,
		document: true,
	});

	const [expandedQuestions, setExpandedQuestions] = useState<
		Record<number, boolean>
	>({});

	const toggleSection = (section: string) => {
		setExpandedSections((prev) => ({
			...prev,
			[section]: !prev[section],
		}));
	};

	const toggleQuestion = (questionId: number) => {
		setExpandedQuestions((prev) => ({
			...prev,
			[questionId]: !prev[questionId],
		}));
	};

	// Calculate research progress
	const completedQuestions = state.plan.filter(
		(q) => q.status === "done",
	).length;
	const totalQuestions = state.plan.length;
	const researchProgress =
		totalQuestions > 0 ? (completedQuestions / totalQuestions) * 100 : 0;

	return (
		<div
			style={{
				display: "flex",
				flexDirection: "column",
				gap: "1rem",
			}}
		>
			{/* Current Phase & Status */}
			<div
				style={{
					background: "white",
					borderRadius: "8px",
					padding: "1.5rem",
					boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
				}}
			>
				<div
					style={{
						fontSize: "0.875rem",
						color: "#718096",
						marginBottom: "0.5rem",
					}}
				>
					Current Phase
				</div>
				<div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
					<div
						style={{
							padding: "0.5rem 1rem",
							background:
								state.phase === "idle"
									? "#e2e8f0"
									: state.phase === "planning"
										? "#fef3c7"
										: state.phase === "searching"
											? "#dbeafe"
											: state.phase === "analyzing"
												? "#e0e7ff"
												: state.phase === "evaluating"
													? "#fce7f3"
													: "#d1fae5",
							color:
								state.phase === "idle"
									? "#718096"
									: state.phase === "planning"
										? "#92400e"
										: state.phase === "searching"
											? "#1e40af"
											: state.phase === "analyzing"
												? "#3730a3"
												: state.phase === "evaluating"
													? "#9f1239"
													: "#065f46",
							borderRadius: "6px",
							fontSize: "1rem",
							fontWeight: "700",
							textTransform: "capitalize",
						}}
					>
						{state.phase}
					</div>
					{state.status && (
						<div
							style={{
								fontSize: "0.875rem",
								color: "#4a5568",
							}}
						>
							{state.status}
						</div>
					)}
				</div>
				{/* Research Progress Bar */}
				{totalQuestions > 0 && state.phase !== "idle" && (
					<div style={{ marginTop: "1rem" }}>
						<div
							style={{
								display: "flex",
								justifyContent: "space-between",
								alignItems: "center",
								marginBottom: "0.5rem",
							}}
						>
							<span
								style={{
									fontSize: "0.75rem",
									color: "#718096",
								}}
							>
								Research Progress
							</span>
							<span
								style={{
									fontSize: "0.75rem",
									fontWeight: "600",
									color: "#2d3748",
								}}
							>
								{completedQuestions}/{totalQuestions} questions
							</span>
						</div>
						<div
							style={{
								height: "0.5rem",
								background: "#e2e8f0",
								borderRadius: "4px",
								overflow: "hidden",
							}}
						>
							<div
								style={{
									width: `${researchProgress}%`,
									height: "100%",
									background: "#48bb78",
									transition: "width 0.3s ease",
								}}
							/>
						</div>
					</div>
				)}
			</div>

			{/* Question */}
			{state.question && (
				<div
					style={{
						background: "white",
						borderRadius: "8px",
						padding: "1.5rem",
						boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
					}}
				>
					<div
						style={{
							fontSize: "0.875rem",
							color: "#718096",
							marginBottom: "0.25rem",
						}}
					>
						Question
					</div>
					<div
						style={{
							fontSize: "1.125rem",
							fontWeight: "bold",
							color: "#2d3748",
						}}
					>
						{state.question}
					</div>
				</div>
			)}

			{/* Confidence Meter */}
			{state.confidence > 0 && (
				<div
					style={{
						background: "white",
						borderRadius: "8px",
						padding: "1.5rem",
						boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
					}}
				>
					<div
						style={{
							fontSize: "0.875rem",
							color: "#718096",
							marginBottom: "0.5rem",
						}}
					>
						Confidence
					</div>
					<div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
						<div
							style={{
								flex: 1,
								height: "1rem",
								background: "#e2e8f0",
								borderRadius: "4px",
								overflow: "hidden",
							}}
						>
							<div
								style={{
									width: `${state.confidence * 100}%`,
									height: "100%",
									background:
										state.confidence > 0.8
											? "#48bb78"
											: state.confidence > 0.5
												? "#ed8936"
												: "#f56565",
									transition: "width 0.3s ease",
								}}
							/>
						</div>
						<div
							style={{
								fontSize: "1.5rem",
								fontWeight: "bold",
								color:
									state.confidence > 0.8
										? "#48bb78"
										: state.confidence > 0.5
											? "#ed8936"
											: "#f56565",
							}}
						>
							{(state.confidence * 100).toFixed(0)}%
						</div>
					</div>
				</div>
			)}

			{/* Research Plan */}
			{state.plan.length > 0 && (
				<div
					style={{
						background: "white",
						borderRadius: "8px",
						boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
						overflow: "hidden",
					}}
				>
					<button
						type="button"
						onClick={() => toggleSection("plan")}
						style={{
							width: "100%",
							display: "flex",
							justifyContent: "space-between",
							alignItems: "center",
							padding: "0.75rem",
							background: "#edf2f7",
							border: "1px solid #e2e8f0",
							borderRadius: "4px",
							cursor: "pointer",
							fontSize: "1rem",
							fontWeight: "600",
							color: "#2d3748",
						}}
					>
						<span>Research Plan ({state.plan.length} questions)</span>
						<span>{expandedSections.plan ? "▼" : "▶"}</span>
					</button>
					{expandedSections.plan && (
						<div
							style={{
								padding: "1rem",
								background: "#f7fafc",
								border: "1px solid #e2e8f0",
								borderTop: "none",
								borderRadius: "0 0 4px 4px",
							}}
						>
							{state.plan.map((item) => (
								<div
									key={item.id}
									style={{
										marginBottom: "0.5rem",
										background: "white",
										borderRadius: "4px",
										border: "1px solid #e2e8f0",
										overflow: "hidden",
									}}
								>
									<button
										type="button"
										onClick={() => toggleQuestion(item.id)}
										style={{
											width: "100%",
											display: "flex",
											gap: "0.75rem",
											padding: "0.75rem",
											background: "white",
											border: "none",
											cursor: "pointer",
											textAlign: "left",
											alignItems: "center",
										}}
									>
										<div
											style={{
												fontSize: "1.25rem",
												color:
													item.status === "done"
														? "#48bb78"
														: item.status === "searching" ||
																item.status === "searched"
															? "#4299e1"
															: "#a0aec0",
												flexShrink: 0,
											}}
										>
											{item.status === "done"
												? "✓"
												: item.status === "searching"
													? "🔍"
													: item.status === "searched"
														? "📊"
														: "⏳"}
										</div>
										<div style={{ flex: 1 }}>
											<div
												style={{
													fontSize: "0.875rem",
													color: "#4a5568",
												}}
											>
												<Markdown content={item.question} />
											</div>
											{item.search_results && (
												<div
													style={{
														fontSize: "0.75rem",
														color: "#718096",
														marginTop: "0.25rem",
													}}
												>
													{item.search_results.results.length} results
												</div>
											)}
										</div>
										{item.search_results && (
											<span
												style={{
													fontSize: "0.875rem",
													color: "#718096",
												}}
											>
												{expandedQuestions[item.id] ? "▼" : "▶"}
											</span>
										)}
									</button>

									{/* Search Results nested inside question */}
									{expandedQuestions[item.id] && item.search_results && (
										<div
											style={{
												padding: "1rem",
												background: "#f7fafc",
												borderTop: "1px solid #e2e8f0",
											}}
										>
											<div
												style={{
													fontSize: "0.75rem",
													color: "#718096",
													marginBottom: "0.5rem",
													fontWeight: "600",
												}}
											>
												Search Type: {item.search_results.type}
											</div>
											{item.search_results.results.map((result, idx) => (
												<div
													key={`${result.chunk_id}-${idx}`}
													style={{
														padding: "0.75rem",
														background: "white",
														borderRadius: "4px",
														marginBottom: "0.5rem",
														border: "1px solid #e2e8f0",
													}}
												>
													<div
														style={{
															display: "flex",
															justifyContent: "space-between",
															marginBottom: "0.5rem",
														}}
													>
														<span
															style={{
																fontSize: "0.875rem",
																fontWeight: "600",
																color: "#2d3748",
															}}
														>
															{result.document_title}
														</span>
														<div style={{ display: "flex", gap: "0.5rem" }}>
															{result.expanded && (
																<span
																	style={{
																		fontSize: "0.75rem",
																		padding: "0.125rem 0.5rem",
																		background: "#bee3f8",
																		color: "#2c5282",
																		borderRadius: "4px",
																	}}
																>
																	Expanded
																</span>
															)}
															<span
																style={{
																	fontSize: "0.875rem",
																	fontWeight: "bold",
																	color:
																		result.score > 0.8
																			? "#48bb78"
																			: result.score > 0.6
																				? "#ed8936"
																				: "#a0aec0",
																}}
															>
																{result.score.toFixed(2)}
															</span>
														</div>
													</div>
													<div
														style={{
															fontSize: "0.875rem",
															color: "#718096",
															lineHeight: "1.4",
														}}
													>
														<Markdown content={`${result.chunk}...`} />
													</div>
												</div>
											))}
										</div>
									)}
								</div>
							))}
						</div>
					)}
				</div>
			)}

			{/* Insights */}
			{state.insights.length > 0 && (
				<div
					style={{
						background: "white",
						borderRadius: "8px",
						boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
						overflow: "hidden",
					}}
				>
					<button
						type="button"
						onClick={() => toggleSection("insights")}
						style={{
							width: "100%",
							display: "flex",
							justifyContent: "space-between",
							alignItems: "center",
							padding: "0.75rem",
							background: "#edf2f7",
							border: "1px solid #e2e8f0",
							borderRadius: "4px",
							cursor: "pointer",
							fontSize: "1rem",
							fontWeight: "600",
							color: "#2d3748",
						}}
					>
						<span>Key Insights ({state.insights.length})</span>
						<span>{expandedSections.insights ? "▼" : "▶"}</span>
					</button>
					{expandedSections.insights && (
						<div
							style={{
								padding: "1rem",
								background: "#f7fafc",
								border: "1px solid #e2e8f0",
								borderTop: "none",
								borderRadius: "0 0 4px 4px",
							}}
						>
							{state.insights.map((insight, idx) => (
								<div
									key={`${insight.summary.substring(0, 30)}-${idx}`}
									style={{
										padding: "0.75rem",
										background: "white",
										borderRadius: "4px",
										marginBottom: "0.5rem",
										border: "1px solid #e2e8f0",
									}}
								>
									<div
										style={{
											display: "flex",
											justifyContent: "space-between",
											marginBottom: "0.5rem",
										}}
									>
										<span
											style={{
												fontSize: "0.75rem",
												padding: "0.125rem 0.5rem",
												background: "#c6f6d5",
												color: "#22543d",
												borderRadius: "4px",
											}}
										>
											{(insight.confidence * 100).toFixed(0)}% confidence
										</span>
										<span
											style={{
												fontSize: "0.75rem",
												color: "#718096",
											}}
										>
											{insight.source_refs?.length || 0} sources
										</span>
									</div>
									<div
										style={{
											fontSize: "0.875rem",
											color: "#2d3748",
											lineHeight: "1.5",
											marginBottom: "0.5rem",
										}}
									>
										<Markdown content={insight.summary} />
									</div>
									{insight.source_refs && insight.source_refs.length > 0 && (
										<div
											style={{
												fontSize: "0.75rem",
												color: "#718096",
												marginTop: "0.5rem",
											}}
										>
											<span style={{ fontWeight: "600" }}>Sources: </span>
											{insight.source_refs.map((ref, refIdx) => (
												<span key={ref.chunk_id}>
													{refIdx > 0 && ", "}
													<span style={{ fontSize: "0.75rem" }}>
														{ref.document_title}
													</span>
												</span>
											))}
										</div>
									)}
								</div>
							))}
						</div>
					)}
				</div>
			)}

			{/* Final Report */}
			{state.final_report && (
				<div
					style={{
						background: "white",
						borderRadius: "8px",
						boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
						overflow: "hidden",
					}}
				>
					<button
						type="button"
						onClick={() => toggleSection("report")}
						style={{
							width: "100%",
							display: "flex",
							justifyContent: "space-between",
							alignItems: "center",
							padding: "0.75rem",
							background: "#edf2f7",
							border: "1px solid #e2e8f0",
							borderRadius: "4px",
							cursor: "pointer",
							fontSize: "1rem",
							fontWeight: "600",
							color: "#2d3748",
						}}
					>
						<span>Final Report</span>
						<span>{expandedSections.report ? "▼" : "▶"}</span>
					</button>
					{expandedSections.report && (
						<div
							style={{
								padding: "1.5rem",
								background: "white",
								border: "1px solid #e2e8f0",
								borderTop: "none",
								borderRadius: "0 0 4px 4px",
							}}
						>
							<h3
								style={{
									fontSize: "1.25rem",
									fontWeight: "600",
									marginBottom: "1rem",
									color: "#2d3748",
								}}
							>
								{state.final_report.title}
							</h3>
							<div style={{ marginBottom: "1.5rem" }}>
								<h4
									style={{
										fontSize: "0.875rem",
										fontWeight: "600",
										color: "#718096",
										marginBottom: "0.5rem",
									}}
								>
									Executive Summary
								</h4>
								<div
									style={{
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									<Markdown content={state.final_report.summary} />
								</div>
							</div>
							<div style={{ marginBottom: "1.5rem" }}>
								<h4
									style={{
										fontSize: "0.875rem",
										fontWeight: "600",
										color: "#718096",
										marginBottom: "0.5rem",
									}}
								>
									Main Findings
								</h4>
								<ul
									style={{
										paddingLeft: "1.5rem",
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									{state.final_report.findings.map((finding, idx) => (
										<li
											key={`finding-${idx}-${finding.substring(0, 30)}`}
											style={{ marginBottom: "0.5rem" }}
										>
											<Markdown content={finding} />
										</li>
									))}
								</ul>
							</div>
							<div style={{ marginBottom: "1.5rem" }}>
								<h4
									style={{
										fontSize: "0.875rem",
										fontWeight: "600",
										color: "#718096",
										marginBottom: "0.5rem",
									}}
								>
									Conclusions
								</h4>
								<ul
									style={{
										paddingLeft: "1.5rem",
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									{state.final_report.conclusions.map((conclusion, idx) => (
										<li
											key={`conclusion-${idx}-${conclusion.substring(0, 30)}`}
											style={{ marginBottom: "0.5rem" }}
										>
											<Markdown content={conclusion} />
										</li>
									))}
								</ul>
							</div>
							<div>
								<h4
									style={{
										fontSize: "0.875rem",
										fontWeight: "600",
										color: "#718096",
										marginBottom: "0.5rem",
									}}
								>
									Citations
								</h4>
								{state.final_report.citations &&
								state.final_report.citations.length > 0 ? (
									<div
										style={{
											display: "flex",
											flexDirection: "column",
											gap: "0.5rem",
										}}
									>
										{state.final_report.citations.map((citation) => (
											<div
												key={citation.document_uri}
												style={{
													padding: "0.5rem",
													background: "#f7fafc",
													borderRadius: "4px",
													border: "1px solid #e2e8f0",
												}}
											>
												<div
													style={{
														fontSize: "0.875rem",
														fontWeight: "600",
														color: "#2d3748",
														marginBottom: "0.25rem",
													}}
												>
													{citation.document_title}
												</div>
												<div
													style={{
														fontSize: "0.75rem",
														color: "#718096",
													}}
												>
													{citation.chunk_ids.length} chunk
													{citation.chunk_ids.length !== 1 ? "s" : ""}{" "}
													referenced
												</div>
											</div>
										))}
									</div>
								) : (
									<div
										style={{
											fontSize: "0.75rem",
											color: "#718096",
											lineHeight: "1.4",
										}}
									>
										{state.final_report.sources?.map((source) => (
											<div key={source} style={{ marginBottom: "0.25rem" }}>
												{source}
											</div>
										))}
									</div>
								)}
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}
