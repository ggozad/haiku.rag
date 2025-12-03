"use client";

import { Markdown } from "@copilotkit/react-ui";
import { useState } from "react";

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
}

interface StateDisplayProps {
	state: ResearchState;
}

export default function StateDisplay({ state }: StateDisplayProps) {
	const [expandedSections, setExpandedSections] = useState<
		Record<string, boolean>
	>({
		questions: true,
		insights: true,
		gaps: true,
		report: true,
	});

	const [expandedQuestions, setExpandedQuestions] = useState<
		Record<string, boolean>
	>({});

	const toggleSection = (section: string) => {
		setExpandedSections((prev) => ({
			...prev,
			[section]: !prev[section],
		}));
	};

	const toggleQuestion = (questionId: string) => {
		setExpandedQuestions((prev) => ({
			...prev,
			[questionId]: !prev[questionId],
		}));
	};

	// Calculate research progress based on iterations
	const researchProgress =
		state.max_iterations > 0
			? (state.iterations / state.max_iterations) * 100
			: 0;
	const confidence = state.last_eval?.confidence || 0;

	return (
		<div
			style={{
				display: "flex",
				flexDirection: "column",
				gap: "1rem",
			}}
		>
			{/* Current Status */}
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
					Research Progress
				</div>
				<div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
					<div
						style={{
							padding: "0.5rem 1rem",
							background:
								state.iterations === 0
									? "#e2e8f0"
									: state.result
										? "#d1fae5"
										: "#dbeafe",
							color:
								state.iterations === 0
									? "#718096"
									: state.result
										? "#065f46"
										: "#1e40af",
							borderRadius: "6px",
							fontSize: "1rem",
							fontWeight: "700",
						}}
					>
						{state.iterations === 0
							? "Ready"
							: state.result
								? "Complete"
								: "Researching"}
					</div>
					{state.iterations > 0 && (
						<div
							style={{
								fontSize: "0.875rem",
								color: "#4a5568",
							}}
						>
							Iteration {state.iterations} of {state.max_iterations}
						</div>
					)}
				</div>
				{/* Research Progress Bar */}
				{state.iterations > 0 && (
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
								Iterations
							</span>
							<span
								style={{
									fontSize: "0.75rem",
									fontWeight: "600",
									color: "#2d3748",
								}}
							>
								{state.iterations}/{state.max_iterations}
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
			{state.context.original_question && (
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
						{state.context.original_question}
					</div>
				</div>
			)}

			{/* Confidence Meter */}
			{confidence > 0 && (
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
									width: `${confidence * 100}%`,
									height: "100%",
									background:
										confidence > 0.8
											? "#48bb78"
											: confidence > 0.5
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
									confidence > 0.8
										? "#48bb78"
										: confidence > 0.5
											? "#ed8936"
											: "#f56565",
							}}
						>
							{(confidence * 100).toFixed(0)}%
						</div>
					</div>
					{state.last_eval?.reasoning && (
						<div
							style={{
								marginTop: "0.75rem",
								fontSize: "0.875rem",
								color: "#4a5568",
								padding: "0.75rem",
								background: "#f7fafc",
								borderRadius: "4px",
							}}
						>
							<Markdown content={state.last_eval.reasoning} />
						</div>
					)}
				</div>
			)}

			{/* Sub-Questions and QA Responses */}
			{(state.context.sub_questions.length > 0 ||
				state.context.qa_responses.length > 0) && (
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
						onClick={() => toggleSection("questions")}
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
						<span>
							Sub-Questions ({state.context.sub_questions.length}) • Answers (
							{state.context.qa_responses.length})
						</span>
						<span>{expandedSections.questions ? "▼" : "▶"}</span>
					</button>
					{expandedSections.questions && (
						<div
							style={{
								padding: "1rem",
								background: "#f7fafc",
								border: "1px solid #e2e8f0",
								borderTop: "none",
								borderRadius: "0 0 4px 4px",
							}}
						>
							{/* Show pending sub_questions */}
							{state.context.sub_questions.map((question, idx) => (
								<div
									key={`pending-${idx}`}
									style={{
										marginBottom: "0.5rem",
										background: "white",
										borderRadius: "4px",
										border: "1px solid #e2e8f0",
										padding: "0.75rem",
										display: "flex",
										gap: "0.75rem",
										alignItems: "center",
									}}
								>
									<div
										style={{
											fontSize: "1.25rem",
											color: "#a0aec0",
											flexShrink: 0,
										}}
									>
										⏳
									</div>
									<div style={{ flex: 1, fontSize: "0.875rem", color: "#4a5568" }}>
										<Markdown content={question} />
									</div>
								</div>
							))}

							{/* Show all qa_responses (each has query + answer) */}
							{state.context.qa_responses.map((qaResponse, idx) => {
								const questionId = `q-${idx}`;

								return (
									<div
										key={questionId}
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
											onClick={() => toggleQuestion(questionId)}
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
													color: "#48bb78",
													flexShrink: 0,
												}}
											>
												✓
											</div>
											<div style={{ flex: 1 }}>
												<div
													style={{
														fontSize: "0.875rem",
														color: "#4a5568",
													}}
												>
													<Markdown content={qaResponse.query} />
												</div>
												<div
													style={{
														fontSize: "0.75rem",
														color: "#718096",
														marginTop: "0.25rem",
													}}
												>
													Confidence: {(qaResponse.confidence * 100).toFixed(0)}%
												</div>
											</div>
											<span
												style={{
													fontSize: "0.875rem",
													color: "#718096",
												}}
											>
												{expandedQuestions[questionId] ? "▼" : "▶"}
											</span>
										</button>

										{/* QA Response nested inside question */}
										{expandedQuestions[questionId] && (
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
													Answer
												</div>
												<div
													style={{
														padding: "0.75rem",
														background: "white",
														borderRadius: "4px",
														border: "1px solid #e2e8f0",
													}}
												>
													<div
														style={{
															fontSize: "0.875rem",
															color: "#2d3748",
															lineHeight: "1.5",
														}}
													>
														<Markdown content={qaResponse.answer} />
													</div>
												</div>

												{/* Citations with visual grounding info */}
												{qaResponse.citations &&
													qaResponse.citations.length > 0 && (
														<div style={{ marginTop: "1rem" }}>
															<div
																style={{
																	fontSize: "0.75rem",
																	color: "#718096",
																	marginBottom: "0.5rem",
																	fontWeight: "600",
																}}
															>
																Citations ({qaResponse.citations.length})
															</div>
															{qaResponse.citations.map((citation, citIdx) => (
																<div
																	key={citIdx}
																	style={{
																		padding: "0.75rem",
																		background: "white",
																		borderRadius: "4px",
																		border: "1px solid #e2e8f0",
																		marginBottom: "0.5rem",
																	}}
																>
																	<div
																		style={{
																			display: "flex",
																			justifyContent: "space-between",
																			alignItems: "flex-start",
																			marginBottom: "0.5rem",
																		}}
																	>
																		<div
																			style={{
																				fontSize: "0.75rem",
																				fontWeight: "600",
																				color: "#2d3748",
																			}}
																		>
																			{citation.document_title ||
																				citation.document_uri}
																		</div>
																		{citation.page_numbers &&
																			citation.page_numbers.length > 0 && (
																				<div
																					style={{
																						fontSize: "0.7rem",
																						color: "#718096",
																						background: "#edf2f7",
																						padding: "0.125rem 0.375rem",
																						borderRadius: "4px",
																					}}
																				>
																					{citation.page_numbers.length === 1
																						? `p. ${citation.page_numbers[0]}`
																						: `pp. ${citation.page_numbers[0]}-${citation.page_numbers[citation.page_numbers.length - 1]}`}
																				</div>
																			)}
																	</div>
																	{citation.headings &&
																		citation.headings.length > 0 && (
																			<div
																				style={{
																					fontSize: "0.7rem",
																					color: "#718096",
																					marginBottom: "0.375rem",
																				}}
																			>
																				{citation.headings.join(" › ")}
																			</div>
																		)}
																	<div
																		style={{
																			fontSize: "0.8rem",
																			color: "#4a5568",
																			lineHeight: "1.4",
																			maxHeight: "4.5rem",
																			overflow: "hidden",
																			textOverflow: "ellipsis",
																		}}
																	>
																		{citation.content.slice(0, 200)}
																		{citation.content.length > 200 && "…"}
																	</div>
																</div>
															))}
														</div>
													)}
											</div>
										)}
									</div>
								);
							})}
						</div>
					)}
				</div>
			)}

			{/* Insights */}
			{state.context.insights.length > 0 && (
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
						<span>Key Insights ({state.context.insights.length})</span>
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
							{state.context.insights.map((insight) => (
								<div
									key={insight.id}
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
												background:
													insight.status === "validated"
														? "#c6f6d5"
														: insight.status === "active"
															? "#bee3f8"
															: "#fed7d7",
												color:
													insight.status === "validated"
														? "#22543d"
														: insight.status === "active"
															? "#2c5282"
															: "#742a2a",
												borderRadius: "4px",
											}}
										>
											{insight.status}
										</span>
										<span
											style={{
												fontSize: "0.75rem",
												color: "#718096",
											}}
										>
											{insight.supporting_sources.length} sources
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
									{insight.notes && (
										<div
											style={{
												fontSize: "0.75rem",
												color: "#718096",
												marginTop: "0.5rem",
												fontStyle: "italic",
											}}
										>
											<Markdown content={insight.notes} />
										</div>
									)}
									{insight.supporting_sources.length > 0 && (
										<div
											style={{
												fontSize: "0.75rem",
												color: "#718096",
												marginTop: "0.5rem",
											}}
										>
											<span style={{ fontWeight: "600" }}>Sources: </span>
											{insight.supporting_sources.map((source, srcIdx) => (
												<span key={`${insight.id}-src-${srcIdx}`}>
													{srcIdx > 0 && ", "}
													{source}
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

			{/* Knowledge Gaps */}
			{state.context.gaps.length > 0 && (
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
						onClick={() => toggleSection("gaps")}
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
						<span>Knowledge Gaps ({state.context.gaps.length})</span>
						<span>{expandedSections.gaps ? "▼" : "▶"}</span>
					</button>
					{expandedSections.gaps && (
						<div
							style={{
								padding: "1rem",
								background: "#f7fafc",
								border: "1px solid #e2e8f0",
								borderTop: "none",
								borderRadius: "0 0 4px 4px",
							}}
						>
							{state.context.gaps.map((gap) => (
								<div
									key={gap.id}
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
											gap: "0.5rem",
											flexWrap: "wrap",
										}}
									>
										<div style={{ display: "flex", gap: "0.5rem" }}>
											<span
												style={{
													fontSize: "0.75rem",
													padding: "0.125rem 0.5rem",
													background:
														gap.severity === "critical"
															? "#fed7d7"
															: gap.severity === "high"
																? "#feebc8"
																: gap.severity === "medium"
																	? "#fef5e7"
																	: "#e6fffa",
													color:
														gap.severity === "critical"
															? "#742a2a"
															: gap.severity === "high"
																? "#7c2d12"
																: gap.severity === "medium"
																	? "#744210"
																	: "#234e52",
													borderRadius: "4px",
													fontWeight: "600",
												}}
											>
												{gap.severity}
											</span>
											{gap.blocking && (
												<span
													style={{
														fontSize: "0.75rem",
														padding: "0.125rem 0.5rem",
														background: "#fed7d7",
														color: "#742a2a",
														borderRadius: "4px",
														fontWeight: "600",
													}}
												>
													Blocking
												</span>
											)}
											{gap.resolved && (
												<span
													style={{
														fontSize: "0.75rem",
														padding: "0.125rem 0.5rem",
														background: "#c6f6d5",
														color: "#22543d",
														borderRadius: "4px",
														fontWeight: "600",
													}}
												>
													Resolved
												</span>
											)}
										</div>
									</div>
									<div
										style={{
											fontSize: "0.875rem",
											color: "#2d3748",
											lineHeight: "1.5",
											marginBottom: "0.5rem",
										}}
									>
										<Markdown content={gap.description} />
									</div>
									{gap.notes && (
										<div
											style={{
												fontSize: "0.75rem",
												color: "#718096",
												marginTop: "0.5rem",
												fontStyle: "italic",
											}}
										>
											<Markdown content={gap.notes} />
										</div>
									)}
									{gap.resolved && gap.resolved_by.length > 0 && (
										<div
											style={{
												fontSize: "0.75rem",
												color: "#718096",
												marginTop: "0.5rem",
											}}
										>
											<span style={{ fontWeight: "600" }}>Resolved by: </span>
											{gap.resolved_by.map((source, srcIdx) => (
												<span key={`${gap.id}-resolved-${srcIdx}`}>
													{srcIdx > 0 && ", "}
													{source}
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
			{state.result && (
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
								{state.result.question}
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
									Summary
								</h4>
								<div
									style={{
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									<Markdown content={state.result.summary} />
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
									Key Findings
								</h4>
								<ul
									style={{
										paddingLeft: "1.5rem",
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									{state.result.findings.map((finding, idx) => (
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
									{state.result.conclusions.map((conclusion, idx) => (
										<li
											key={`conclusion-${idx}-${conclusion.substring(0, 30)}`}
											style={{ marginBottom: "0.5rem" }}
										>
											<Markdown content={conclusion} />
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
									Methodology
								</h4>
								<div
									style={{
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									<Markdown content={state.result.methodology} />
								</div>
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
									Insights Used ({state.result.insights_used.length})
								</h4>
								<div
									style={{
										display: "flex",
										flexDirection: "column",
										gap: "0.5rem",
									}}
								>
									{state.result.insights_used.map((insightId, idx) => {
										const insight = state.context.insights.find(
											(i) => i.id === insightId,
										);
										return (
											<div
												key={`insight-${idx}-${insightId}`}
												style={{
													padding: "0.5rem",
													background: "#f7fafc",
													borderRadius: "4px",
													border: "1px solid #e2e8f0",
												}}
											>
												{insight ? (
													<div
														style={{
															fontSize: "0.875rem",
															color: "#2d3748",
															lineHeight: "1.4",
														}}
													>
														<Markdown content={insight.summary} />
													</div>
												) : (
													<div
														style={{
															fontSize: "0.875rem",
															color: "#718096",
														}}
													>
														Insight ID: {insightId}
													</div>
												)}
											</div>
										);
									})}
								</div>
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}
