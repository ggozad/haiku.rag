"use client";

import { useState } from "react";

interface ResearchState {
	question: string;
	phase: string;
	status: string;
	plan: Array<{
		id: number;
		question: string;
		status: string;
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

interface StateDisplayProps {
	state: ResearchState;
}

export default function StateDisplay({ state }: StateDisplayProps) {
	const [expandedSections, setExpandedSections] = useState<
		Record<string, boolean>
	>({
		plan: true,
		search: true,
		insights: true,
		report: true,
	});

	const toggleSection = (section: string) => {
		setExpandedSections((prev) => ({
			...prev,
			[section]: !prev[section],
		}));
	};

	// Phase indicator
	const phases = [
		"idle",
		"planning",
		"searching",
		"analyzing",
		"evaluating",
		"done",
	];
	const currentPhaseIndex = phases.indexOf(state.phase);

	return (
		<div
			style={{
				background: "white",
				borderRadius: "8px",
				padding: "1.5rem",
				boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
				marginTop: "2rem",
			}}
		>
			<h2
				style={{
					fontSize: "1.5rem",
					fontWeight: "600",
					marginBottom: "1rem",
					color: "#2d3748",
				}}
			>
				Research State
			</h2>

			{/* Phase Progress */}
			<div style={{ marginBottom: "2rem" }}>
				<div
					style={{
						fontSize: "0.875rem",
						color: "#718096",
						marginBottom: "0.5rem",
					}}
				>
					Progress
				</div>
				<div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
					{phases.slice(1).map((phase, idx) => (
						<div key={phase} style={{ display: "flex", alignItems: "center" }}>
							<div
								style={{
									padding: "0.25rem 0.75rem",
									background:
										idx < currentPhaseIndex
											? "#48bb78"
											: idx === currentPhaseIndex
												? "#4299e1"
												: "#e2e8f0",
									color:
										idx < currentPhaseIndex || idx === currentPhaseIndex
											? "white"
											: "#718096",
									borderRadius: "4px",
									fontSize: "0.75rem",
									fontWeight: "600",
									textTransform: "capitalize",
								}}
							>
								{phase}
							</div>
							{idx < phases.length - 2 && (
								<div
									style={{
										width: "1rem",
										height: "2px",
										background: idx < currentPhaseIndex ? "#48bb78" : "#e2e8f0",
										margin: "0 0.25rem",
									}}
								/>
							)}
						</div>
					))}
				</div>
			</div>

			{/* Question */}
			{state.question && (
				<div
					style={{
						padding: "1rem",
						background: "#f7fafc",
						borderRadius: "4px",
						border: "1px solid #e2e8f0",
						marginBottom: "1rem",
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
						padding: "1rem",
						background: "#f7fafc",
						borderRadius: "4px",
						border: "1px solid #e2e8f0",
						marginBottom: "1rem",
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
				<div style={{ marginBottom: "1rem" }}>
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
										padding: "0.75rem",
										background: "white",
										borderRadius: "4px",
										marginBottom: "0.5rem",
										border: "1px solid #e2e8f0",
										display: "flex",
										gap: "0.75rem",
									}}
								>
									<div
										style={{
											fontSize: "1.25rem",
											color:
												item.status === "done"
													? "#48bb78"
													: item.status === "searching"
														? "#4299e1"
														: "#a0aec0",
										}}
									>
										{item.status === "done"
											? "✓"
											: item.status === "searching"
												? "🔍"
												: "⏳"}
									</div>
									<div style={{ flex: 1 }}>
										<div
											style={{
												fontSize: "0.875rem",
												color: "#4a5568",
											}}
										>
											{item.question}
										</div>
									</div>
								</div>
							))}
						</div>
					)}
				</div>
			)}

			{/* Current Search Results */}
			{state.current_search && (
				<div style={{ marginBottom: "1rem" }}>
					<button
						type="button"
						onClick={() => toggleSection("search")}
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
							Search Results: {state.current_search.query.substring(0, 50)}...
						</span>
						<span>{expandedSections.search ? "▼" : "▶"}</span>
					</button>
					{expandedSections.search && (
						<div
							style={{
								padding: "1rem",
								background: "#f7fafc",
								border: "1px solid #e2e8f0",
								borderTop: "none",
								borderRadius: "0 0 4px 4px",
							}}
						>
							{state.current_search.results && (
								<div>
									<div
										style={{
											fontSize: "0.875rem",
											color: "#718096",
											marginBottom: "0.5rem",
										}}
									>
										Type: {state.current_search.type} |{" "}
										{state.current_search.results.length} results
									</div>
									{state.current_search.results.map((result, idx) => (
										<div
											key={`${result.source}-${idx}`}
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
														color: "#4a5568",
													}}
												>
													{result.source}
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
												{result.chunk}...
											</div>
										</div>
									))}
								</div>
							)}
						</div>
					)}
				</div>
			)}

			{/* Insights */}
			{state.insights.length > 0 && (
				<div style={{ marginBottom: "1rem" }}>
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
											{insight.sources.length} sources
										</span>
									</div>
									<div
										style={{
											fontSize: "0.875rem",
											color: "#2d3748",
											lineHeight: "1.5",
										}}
									>
										{insight.summary}
									</div>
								</div>
							))}
						</div>
					)}
				</div>
			)}

			{/* Final Report */}
			{state.final_report && (
				<div>
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
								<p
									style={{
										fontSize: "0.875rem",
										color: "#4a5568",
										lineHeight: "1.6",
									}}
								>
									{state.final_report.summary}
								</p>
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
									{state.final_report.findings.map((finding) => (
										<li key={finding} style={{ marginBottom: "0.5rem" }}>
											{finding}
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
									{state.final_report.conclusions.map((conclusion) => (
										<li key={conclusion} style={{ marginBottom: "0.5rem" }}>
											{conclusion}
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
									Sources
								</h4>
								<div
									style={{
										fontSize: "0.75rem",
										color: "#718096",
										lineHeight: "1.4",
									}}
								>
									{state.final_report.sources.map((source) => (
										<div key={source} style={{ marginBottom: "0.25rem" }}>
											{source}
										</div>
									))}
								</div>
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}
