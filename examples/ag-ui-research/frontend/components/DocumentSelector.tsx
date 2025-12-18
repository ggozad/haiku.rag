"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

interface Document {
	id: string;
	title: string;
	uri: string;
}

interface DocumentSelectorProps {
	selected: string[];
	onChange: (ids: string[]) => void;
	disabled: boolean;
}

export default function DocumentSelector({
	selected,
	onChange,
	disabled,
}: DocumentSelectorProps) {
	const [documents, setDocuments] = useState<Document[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [expanded, setExpanded] = useState(true);
	const [searchQuery, setSearchQuery] = useState("");

	useEffect(() => {
		const fetchDocuments = async () => {
			try {
				const response = await fetch(
					`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/documents`,
				);
				if (!response.ok) {
					throw new Error("Failed to fetch documents");
				}
				const data = await response.json();
				const docs = data.documents || [];
				setDocuments(docs);
				// Select all documents by default if none are selected
				if (selected.length === 0 && docs.length > 0) {
					onChange(docs.map((d: Document) => d.id));
				}
				setError(null);
			} catch (err) {
				setError(err instanceof Error ? err.message : "Unknown error");
			} finally {
				setLoading(false);
			}
		};

		fetchDocuments();
	}, []);

	const filteredDocuments = useMemo(() => {
		if (!searchQuery.trim()) return documents;
		const query = searchQuery.toLowerCase();
		return documents.filter(
			(doc) =>
				(doc.title || "").toLowerCase().includes(query) ||
				(doc.uri || "").toLowerCase().includes(query),
		);
	}, [documents, searchQuery]);

	const handleToggle = useCallback(
		(id: string) => {
			if (disabled) return;
			if (selected.includes(id)) {
				onChange(selected.filter((s) => s !== id));
			} else {
				onChange([...selected, id]);
			}
		},
		[selected, onChange, disabled],
	);

	const handleSelectAll = useCallback(() => {
		if (disabled) return;
		const filteredIds = filteredDocuments.map((d) => d.id);
		const allFilteredSelected = filteredIds.every((id) =>
			selected.includes(id),
		);
		if (allFilteredSelected) {
			// Deselect all filtered documents
			onChange(selected.filter((id) => !filteredIds.includes(id)));
		} else {
			// Select all filtered documents (add to existing selection)
			const newSelection = [...new Set([...selected, ...filteredIds])];
			onChange(newSelection);
		}
	}, [selected, filteredDocuments, onChange, disabled]);

	const selectedCount = selected.length;
	const totalCount = documents.length;
	const filterActive = selectedCount > 0 && selectedCount < totalCount;

	return (
		<div
			style={{
				background: "white",
				borderRadius: "8px",
				boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
				overflow: "hidden",
				opacity: disabled ? 0.6 : 1,
				transition: "opacity 0.2s",
			}}
		>
			<button
				type="button"
				onClick={() => setExpanded(!expanded)}
				style={{
					width: "100%",
					display: "flex",
					justifyContent: "space-between",
					alignItems: "center",
					padding: "0.75rem",
					background: filterActive ? "#ebf8ff" : "#edf2f7",
					border: filterActive ? "1px solid #90cdf4" : "1px solid #e2e8f0",
					borderRadius: expanded ? "8px 8px 0 0" : "8px",
					cursor: "pointer",
					fontSize: "0.875rem",
					fontWeight: "600",
					color: filterActive ? "#2b6cb0" : "#2d3748",
				}}
			>
				<span>
					Document Filter
					{filterActive && ` (${selectedCount}/${totalCount})`}
					{!filterActive && selectedCount === 0 && " (All)"}
				</span>
				<span>{expanded ? "▼" : "▶"}</span>
			</button>

			{expanded && (
				<div
					style={{
						padding: "0.75rem",
						background: "#f7fafc",
						borderLeft: "1px solid #e2e8f0",
						borderRight: "1px solid #e2e8f0",
						borderBottom: "1px solid #e2e8f0",
						borderRadius: "0 0 8px 8px",
					}}
				>
					{loading && (
						<div
							style={{
								padding: "1rem",
								textAlign: "center",
								color: "#718096",
								fontSize: "0.875rem",
							}}
						>
							Loading documents...
						</div>
					)}

					{error && (
						<div
							style={{
								padding: "0.75rem",
								background: "#fed7d7",
								color: "#c53030",
								borderRadius: "4px",
								fontSize: "0.875rem",
							}}
						>
							{error}
						</div>
					)}

					{!loading && !error && documents.length === 0 && (
						<div
							style={{
								padding: "1rem",
								textAlign: "center",
								color: "#718096",
								fontSize: "0.875rem",
							}}
						>
							No documents in database
						</div>
					)}

					{!loading && !error && documents.length > 0 && (
						<>
							{/* Search Input */}
							<div style={{ marginBottom: "0.5rem" }}>
								<input
									type="text"
									placeholder="Search by title or URI..."
									value={searchQuery}
									onChange={(e) => setSearchQuery(e.target.value)}
									disabled={disabled}
									style={{
										width: "100%",
										padding: "0.5rem 0.75rem",
										fontSize: "0.875rem",
										border: "1px solid #e2e8f0",
										borderRadius: "4px",
										background: disabled ? "#f7fafc" : "white",
										color: disabled ? "#a0aec0" : "#2d3748",
										outline: "none",
									}}
								/>
							</div>

							{/* Select All / Clear */}
							<div
								style={{
									marginBottom: "0.5rem",
									paddingBottom: "0.5rem",
									borderBottom: "1px solid #e2e8f0",
									display: "flex",
									alignItems: "center",
									justifyContent: "space-between",
								}}
							>
								<div>
									<button
										type="button"
										onClick={handleSelectAll}
										disabled={disabled}
										style={{
											padding: "0.375rem 0.75rem",
											fontSize: "0.75rem",
											background: disabled ? "#e2e8f0" : "#4299e1",
											color: disabled ? "#a0aec0" : "white",
											border: "none",
											borderRadius: "4px",
											cursor: disabled ? "not-allowed" : "pointer",
										}}
									>
										{filteredDocuments.every((d) => selected.includes(d.id))
											? "Clear Visible"
											: "Select Visible"}
									</button>
									<span
										style={{
											marginLeft: "0.75rem",
											fontSize: "0.75rem",
											color: "#718096",
										}}
									>
										{selectedCount} of {totalCount} selected
									</span>
								</div>
								{searchQuery && (
									<span
										style={{
											fontSize: "0.75rem",
											color: "#718096",
										}}
									>
										Showing {filteredDocuments.length} of {totalCount}
									</span>
								)}
							</div>

							{/* Document List */}
							<div
								style={{
									maxHeight: "200px",
									overflowY: "auto",
									display: "flex",
									flexDirection: "column",
									gap: "0.25rem",
								}}
							>
								{filteredDocuments.map((doc) => {
									const isSelected = selected.includes(doc.id);
									return (
										<label
											key={doc.id}
											style={{
												display: "flex",
												alignItems: "center",
												gap: "0.5rem",
												padding: "0.5rem",
												background: isSelected ? "#ebf8ff" : "white",
												border: isSelected
													? "1px solid #90cdf4"
													: "1px solid #e2e8f0",
												borderRadius: "4px",
												cursor: disabled ? "not-allowed" : "pointer",
												transition: "all 0.15s",
											}}
										>
											<input
												type="checkbox"
												checked={isSelected}
												onChange={() => handleToggle(doc.id)}
												disabled={disabled}
												style={{
													width: "1rem",
													height: "1rem",
													cursor: disabled ? "not-allowed" : "pointer",
												}}
											/>
											<div style={{ flex: 1, minWidth: 0 }}>
												{doc.title && (
													<div
														style={{
															fontSize: "0.875rem",
															fontWeight: isSelected ? "600" : "400",
															color: "#2d3748",
															whiteSpace: "nowrap",
															overflow: "hidden",
															textOverflow: "ellipsis",
														}}
													>
														{doc.title}
													</div>
												)}
												<div
													style={{
														fontSize: doc.title ? "0.7rem" : "0.875rem",
														fontWeight: doc.title
															? "400"
															: isSelected
																? "600"
																: "400",
														color: doc.title ? "#718096" : "#2d3748",
														whiteSpace: "nowrap",
														overflow: "hidden",
														textOverflow: "ellipsis",
													}}
												>
													{doc.uri}
												</div>
											</div>
										</label>
									);
								})}
							</div>
						</>
					)}

					{disabled && (
						<div
							style={{
								marginTop: "0.5rem",
								padding: "0.5rem",
								background: "#fef5e7",
								border: "1px solid #f6ad55",
								borderRadius: "4px",
								fontSize: "0.75rem",
								color: "#744210",
								textAlign: "center",
							}}
						>
							Filter locked during research
						</div>
					)}
				</div>
			)}
		</div>
	);
}
