"use client";

import { Markdown } from "@copilotkit/react-ui";
import { useCallback, useState } from "react";

interface VisualGroundingState {
	isOpen: boolean;
	chunkId: string | null;
	images: string[];
	loading: boolean;
	error: string | null;
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
}

interface StateDisplayProps {
	state: ResearchState;
}

export default function StateDisplay({ state }: StateDisplayProps) {
	const [expandedSections, setExpandedSections] = useState<
		Record<string, boolean>
	>({
		questions: true,
		report: true,
	});

	const [expandedQuestions, setExpandedQuestions] = useState<
		Record<string, boolean>
	>({});

	const [visualGrounding, setVisualGrounding] = useState<VisualGroundingState>({
		isOpen: false,
		chunkId: null,
		images: [],
		loading: false,
		error: null,
	});

	const fetchVisualGrounding = useCallback(async (chunkId: string) => {
		setVisualGrounding({
			isOpen: true,
			chunkId,
			images: [],
			loading: true,
			error: null,
		});

		try {
			const response = await fetch(
				`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/visualize/${chunkId}`,
			);
			const data = await response.json();

			if (!response.ok) {
				throw new Error(data.error || "Failed to fetch visual grounding");
			}

			setVisualGrounding((prev) => ({
				...prev,
				images: data.images || [],
				loading: false,
				error: data.images?.length === 0 ? data.message : null,
			}));
		} catch (err) {
			setVisualGrounding((prev) => ({
				...prev,
				loading: false,
				error: err instanceof Error ? err.message : "Unknown error",
			}));
		}
	}, []);

	const closeVisualGrounding = useCallback(() => {
		setVisualGrounding({
			isOpen: false,
			chunkId: null,
			images: [],
			loading: false,
			error: null,
		});
	}, []);

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
	const confidence = state.last_eval?.confidence_score || 0;

	return (
		<div
			style={{
				display: "flex",
				flexDirection: "column",
				gap: "1rem",
			}}
		>
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

			{/* Research Progress - only show when research is in progress (not when complete) */}
			{(state.iterations > 0 || state.current_activity) && !state.result && (
				<div
					style={{
						background: "white",
						borderRadius: "8px",
						padding: "1.5rem",
						boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
					}}
				>
					{/* Current Activity */}
					{state.current_activity && (
						<div
							style={{
								padding: "0.75rem",
								background: "#ebf8ff",
								borderRadius: "6px",
								border: "1px solid #90cdf4",
								marginBottom: state.iterations > 0 ? "1rem" : 0,
							}}
						>
							<div
								style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
							>
								<span>⏳</span>
								<span style={{ fontWeight: "600", color: "#2b6cb0" }}>
									{state.current_activity
										.replace(/_/g, " ")
										.replace(/\b\w/g, (c) => c.toUpperCase())}
								</span>
							</div>
							{state.current_activity_message && (
								<div
									style={{
										fontSize: "0.875rem",
										color: "#4299e1",
										marginTop: "0.25rem",
										marginLeft: "1.5rem",
									}}
								>
									{state.current_activity_message}
								</div>
							)}
						</div>
					)}
					{/* Iteration Progress Bar */}
					{state.iterations > 0 && (
						<div>
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
										background: state.result ? "#48bb78" : "#4299e1",
										transition: "width 0.3s ease",
									}}
								/>
							</div>
						</div>
					)}
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

			{/* Answers */}
			{state.context.qa_responses.length > 0 && (
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
						<span>Answers ({state.context.qa_responses.length})</span>
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
													Confidence: {(qaResponse.confidence * 100).toFixed(0)}
													%
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
																	<button
																		type="button"
																		onClick={() =>
																			fetchVisualGrounding(citation.chunk_id)
																		}
																		style={{
																			marginTop: "0.5rem",
																			padding: "0.25rem 0.5rem",
																			fontSize: "0.7rem",
																			background: "#4299e1",
																			color: "white",
																			border: "none",
																			borderRadius: "4px",
																			cursor: "pointer",
																		}}
																	>
																		📍 View in Document
																	</button>
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
								{state.result.title}
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
									<Markdown content={state.result.executive_summary} />
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
									{state.result.main_findings.map((finding, idx) => (
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
							{state.result.recommendations.length > 0 && (
								<div style={{ marginBottom: "1.5rem" }}>
									<h4
										style={{
											fontSize: "0.875rem",
											fontWeight: "600",
											color: "#718096",
											marginBottom: "0.5rem",
										}}
									>
										Recommendations
									</h4>
									<ul
										style={{
											paddingLeft: "1.5rem",
											fontSize: "0.875rem",
											color: "#4a5568",
											lineHeight: "1.6",
										}}
									>
										{state.result.recommendations.map((rec, idx) => (
											<li
												key={`rec-${idx}-${rec.substring(0, 30)}`}
												style={{ marginBottom: "0.5rem" }}
											>
												<Markdown content={rec} />
											</li>
										))}
									</ul>
								</div>
							)}
							{state.result.limitations.length > 0 && (
								<div style={{ marginBottom: "1.5rem" }}>
									<h4
										style={{
											fontSize: "0.875rem",
											fontWeight: "600",
											color: "#718096",
											marginBottom: "0.5rem",
										}}
									>
										Limitations
									</h4>
									<ul
										style={{
											paddingLeft: "1.5rem",
											fontSize: "0.875rem",
											color: "#4a5568",
											lineHeight: "1.6",
										}}
									>
										{state.result.limitations.map((lim, idx) => (
											<li
												key={`lim-${idx}-${lim.substring(0, 30)}`}
												style={{ marginBottom: "0.5rem" }}
											>
												<Markdown content={lim} />
											</li>
										))}
									</ul>
								</div>
							)}
							<div>
								<h4
									style={{
										fontSize: "0.875rem",
										fontWeight: "600",
										color: "#718096",
										marginBottom: "0.5rem",
									}}
								>
									Sources
								</h4>
								<div
									style={{
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									<Markdown content={state.result.sources_summary} />
								</div>
							</div>
						</div>
					)}
				</div>
			)}

			{/* Visual Grounding Modal */}
			{visualGrounding.isOpen && (
				<div
					style={{
						position: "fixed",
						top: 0,
						left: 0,
						right: 0,
						bottom: 0,
						background: "rgba(0, 0, 0, 0.75)",
						display: "flex",
						alignItems: "center",
						justifyContent: "center",
						zIndex: 1000,
					}}
					onClick={closeVisualGrounding}
				>
					<div
						style={{
							background: "white",
							borderRadius: "8px",
							padding: "1.5rem",
							maxWidth: "90vw",
							maxHeight: "90vh",
							overflow: "auto",
							position: "relative",
						}}
						onClick={(e) => e.stopPropagation()}
					>
						<button
							type="button"
							onClick={closeVisualGrounding}
							style={{
								position: "absolute",
								top: "0.5rem",
								right: "0.5rem",
								background: "#e53e3e",
								color: "white",
								border: "none",
								borderRadius: "50%",
								width: "2rem",
								height: "2rem",
								cursor: "pointer",
								fontSize: "1rem",
							}}
						>
							✕
						</button>
						<h3
							style={{
								margin: "0 0 1rem 0",
								fontSize: "1.125rem",
								color: "#2d3748",
							}}
						>
							Visual Grounding
						</h3>
						{visualGrounding.loading && (
							<div
								style={{
									padding: "2rem",
									textAlign: "center",
									color: "#718096",
								}}
							>
								Loading...
							</div>
						)}
						{visualGrounding.error && (
							<div
								style={{
									padding: "1rem",
									background: "#fed7d7",
									color: "#c53030",
									borderRadius: "4px",
								}}
							>
								{visualGrounding.error}
							</div>
						)}
						{!visualGrounding.loading &&
							!visualGrounding.error &&
							visualGrounding.images.length > 0 && (
								<div
									style={{
										display: "flex",
										flexDirection: "column",
										gap: "1rem",
									}}
								>
									{visualGrounding.images.map((img, idx) => (
										<div key={idx}>
											<div
												style={{
													fontSize: "0.75rem",
													color: "#718096",
													marginBottom: "0.5rem",
												}}
											>
												Page {idx + 1} of {visualGrounding.images.length}
											</div>
											<img
												src={`data:image/png;base64,${img}`}
												alt={`Page ${idx + 1}`}
												style={{
													maxWidth: "100%",
													border: "1px solid #e2e8f0",
													borderRadius: "4px",
												}}
											/>
										</div>
									))}
								</div>
							)}
					</div>
				</div>
			)}
		</div>
	);
}
