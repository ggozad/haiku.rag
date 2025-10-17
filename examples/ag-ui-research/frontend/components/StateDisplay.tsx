"use client";

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

interface StateDisplayProps {
	state: ResearchState;
}

export default function StateDisplay({ state }: StateDisplayProps) {
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
			<p
				style={{
					color: "#4a5568",
					lineHeight: "1.6",
					marginBottom: "1rem",
					fontSize: "0.875rem",
				}}
			>
				This state is shared between the research agent and the frontend via the
				AG-UI protocol.
			</p>

			<div
				style={{
					display: "grid",
					gap: "1rem",
					marginTop: "1rem",
				}}
			>
				<div
					style={{
						padding: "1rem",
						background: "#f7fafc",
						borderRadius: "4px",
						border: "1px solid #e2e8f0",
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
						{state.question || "No question yet"}
					</div>
				</div>

				<div
					style={{
						display: "grid",
						gridTemplateColumns: "repeat(2, 1fr)",
						gap: "1rem",
					}}
				>
					<div
						style={{
							padding: "1rem",
							background: "#f7fafc",
							borderRadius: "4px",
							border: "1px solid #e2e8f0",
						}}
					>
						<div
							style={{
								fontSize: "0.875rem",
								color: "#718096",
								marginBottom: "0.25rem",
							}}
						>
							Status
						</div>
						<div
							style={{
								fontSize: "1.125rem",
								fontWeight: "bold",
								color: state.status === "idle" ? "#718096" : "#38a169",
							}}
						>
							{state.status}
						</div>
					</div>

					<div
						style={{
							padding: "1rem",
							background: "#f7fafc",
							borderRadius: "4px",
							border: "1px solid #e2e8f0",
						}}
					>
						<div
							style={{
								fontSize: "0.875rem",
								color: "#718096",
								marginBottom: "0.25rem",
							}}
						>
							Confidence
						</div>
						<div
							style={{
								fontSize: "1.5rem",
								fontWeight: "bold",
								color:
									state.confidence > 0.8
										? "#38a169"
										: state.confidence > 0.5
											? "#d69e2e"
											: "#e53e3e",
							}}
						>
							{(state.confidence * 100).toFixed(0)}%
						</div>
					</div>
				</div>

				<div
					style={{
						display: "grid",
						gridTemplateColumns: "repeat(3, 1fr)",
						gap: "1rem",
					}}
				>
					<div
						style={{
							padding: "1rem",
							background: "#f7fafc",
							borderRadius: "4px",
							border: "1px solid #e2e8f0",
						}}
					>
						<div
							style={{
								fontSize: "0.875rem",
								color: "#718096",
								marginBottom: "0.25rem",
							}}
						>
							Progress
						</div>
						<div
							style={{
								fontSize: "1.125rem",
								fontWeight: "bold",
								color: "#2d3748",
							}}
						>
							{state.current_iteration} / {state.max_iterations}
						</div>
					</div>

					<div
						style={{
							padding: "1rem",
							background: "#f7fafc",
							borderRadius: "4px",
							border: "1px solid #e2e8f0",
						}}
					>
						<div
							style={{
								fontSize: "0.875rem",
								color: "#718096",
								marginBottom: "0.25rem",
							}}
						>
							Plan Items
						</div>
						<div
							style={{
								fontSize: "1.125rem",
								fontWeight: "bold",
								color: "#2d3748",
							}}
						>
							{state.plan.length}
						</div>
					</div>

					<div
						style={{
							padding: "1rem",
							background: "#f7fafc",
							borderRadius: "4px",
							border: "1px solid #e2e8f0",
						}}
					>
						<div
							style={{
								fontSize: "0.875rem",
								color: "#718096",
								marginBottom: "0.25rem",
							}}
						>
							Findings
						</div>
						<div
							style={{
								fontSize: "1.125rem",
								fontWeight: "bold",
								color: "#2d3748",
							}}
						>
							{state.findings.length}
						</div>
					</div>
				</div>

				<div
					style={{
						padding: "1rem",
						background: "#f7fafc",
						borderRadius: "4px",
						border: "1px solid #e2e8f0",
					}}
				>
					<div
						style={{
							fontSize: "0.875rem",
							color: "#718096",
							marginBottom: "0.25rem",
						}}
					>
						Final Report
					</div>
					<div
						style={{
							fontSize: "1.125rem",
							fontWeight: "bold",
							color: state.final_report ? "#38a169" : "#a0aec0",
						}}
					>
						{state.final_report ? "Ready" : "Not ready"}
					</div>
				</div>
			</div>
		</div>
	);
}
