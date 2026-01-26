"use client";

import { useCallback, useEffect, useId, useState } from "react";

interface Document {
	id: string;
	title: string | null;
	uri: string | null;
}

interface DocumentFilterProps {
	isOpen: boolean;
	onClose: () => void;
	selected: string[];
	onApply: (selected: string[]) => void;
}

function FilterIcon() {
	return (
		<svg
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="1.5"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
		</svg>
	);
}

export default function DocumentFilter({
	isOpen,
	onClose,
	selected,
	onApply,
}: DocumentFilterProps) {
	const titleId = useId();
	const [documents, setDocuments] = useState<Document[]>([]);
	const [loading, setLoading] = useState(true);
	const [searchTerm, setSearchTerm] = useState("");
	const [localSelected, setLocalSelected] = useState<Set<string>>(
		new Set(selected),
	);

	// Reset local state when modal opens
	useEffect(() => {
		if (isOpen) {
			setLocalSelected(new Set(selected));
			setSearchTerm("");
		}
	}, [isOpen, selected]);

	// Fetch documents when modal opens
	useEffect(() => {
		if (isOpen && documents.length === 0) {
			setLoading(true);
			fetch("/api/documents")
				.then((res) => res.json())
				.then((data) => {
					setDocuments(data.documents || []);
					setLoading(false);
				})
				.catch(() => {
					setLoading(false);
				});
		}
	}, [isOpen, documents.length]);

	const handleKeyDown = useCallback(
		(e: React.KeyboardEvent) => {
			if (e.key === "Escape") {
				onClose();
			}
		},
		[onClose],
	);

	const toggleDocument = (displayName: string) => {
		setLocalSelected((prev) => {
			const next = new Set(prev);
			if (next.has(displayName)) {
				next.delete(displayName);
			} else {
				next.add(displayName);
			}
			return next;
		});
	};

	const handleApply = () => {
		onApply(Array.from(localSelected));
		onClose();
	};

	const handleClearAll = () => {
		setLocalSelected(new Set());
	};

	const getDisplayName = (doc: Document) => doc.title || doc.uri || doc.id;

	const filteredDocuments = documents.filter((doc) => {
		if (!searchTerm) return true;
		const displayName = getDisplayName(doc).toLowerCase();
		return displayName.includes(searchTerm.toLowerCase());
	});

	if (!isOpen) {
		return null;
	}

	return (
		<>
			<style>{`
				.filter-modal-overlay {
					position: fixed;
					top: 0;
					left: 0;
					right: 0;
					bottom: 0;
					background: rgba(0, 0, 0, 0.5);
					display: flex;
					align-items: center;
					justify-content: center;
					z-index: 1000;
				}
				.filter-modal {
					background: white;
					border-radius: 12px;
					padding: 1.5rem;
					width: 90%;
					max-width: 500px;
					max-height: 80vh;
					display: flex;
					flex-direction: column;
					position: relative;
					box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
				}
				.filter-modal-header {
					display: flex;
					align-items: center;
					gap: 0.75rem;
					margin-bottom: 0.5rem;
				}
				.filter-modal-icon {
					display: flex;
					align-items: center;
					justify-content: center;
					width: 40px;
					height: 40px;
					border-radius: 10px;
					background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
					color: #d97706;
				}
				.filter-modal-title {
					margin: 0;
					font-size: 1.25rem;
					font-weight: 600;
					color: #1e293b;
				}
				.filter-modal-description {
					margin: 0 0 1rem 0;
					font-size: 0.875rem;
					color: #64748b;
					line-height: 1.5;
				}
				.filter-search {
					width: 100%;
					padding: 0.625rem 0.875rem;
					font-size: 0.875rem;
					border: 1px solid #e2e8f0;
					border-radius: 8px;
					margin-bottom: 0.75rem;
					outline: none;
					transition: border-color 0.15s;
				}
				.filter-search:focus {
					border-color: #3b82f6;
					box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
				}
				.filter-list {
					flex: 1;
					min-height: 200px;
					max-height: 300px;
					overflow-y: auto;
					border: 1px solid #e2e8f0;
					border-radius: 8px;
					background: #f8fafc;
				}
				.filter-item {
					display: flex;
					align-items: center;
					gap: 0.75rem;
					padding: 0.625rem 0.875rem;
					cursor: pointer;
					transition: background 0.1s;
					border-bottom: 1px solid #e2e8f0;
				}
				.filter-item:last-child {
					border-bottom: none;
				}
				.filter-item:hover {
					background: #f1f5f9;
				}
				.filter-item input[type="checkbox"] {
					width: 16px;
					height: 16px;
					cursor: pointer;
					accent-color: #3b82f6;
				}
				.filter-item-label {
					flex: 1;
					font-size: 0.875rem;
					color: #334155;
					overflow: hidden;
					text-overflow: ellipsis;
					white-space: nowrap;
				}
				.filter-loading, .filter-empty {
					display: flex;
					align-items: center;
					justify-content: center;
					height: 100px;
					color: #94a3b8;
					font-size: 0.875rem;
				}
				.filter-footer {
					display: flex;
					justify-content: space-between;
					align-items: center;
					margin-top: 1rem;
					padding-top: 1rem;
					border-top: 1px solid #e2e8f0;
				}
				.filter-count {
					font-size: 0.75rem;
					color: #64748b;
				}
				.filter-count strong {
					color: #3b82f6;
				}
				.filter-buttons {
					display: flex;
					gap: 0.5rem;
				}
				.filter-btn {
					padding: 0.5rem 1rem;
					font-size: 0.875rem;
					font-weight: 500;
					border-radius: 6px;
					cursor: pointer;
					transition: all 0.15s;
				}
				.filter-btn-secondary {
					background: white;
					color: #475569;
					border: 1px solid #e2e8f0;
				}
				.filter-btn-secondary:hover {
					background: #f8fafc;
					border-color: #cbd5e1;
				}
				.filter-btn-primary {
					background: #3b82f6;
					color: white;
					border: 1px solid #3b82f6;
				}
				.filter-btn-primary:hover {
					background: #2563eb;
					border-color: #2563eb;
				}
				.filter-btn-clear {
					background: transparent;
					color: #ef4444;
					border: none;
					padding: 0.5rem;
					font-size: 0.75rem;
				}
				.filter-btn-clear:hover {
					text-decoration: underline;
				}
			`}</style>
			<div
				className="filter-modal-overlay"
				onClick={onClose}
				onKeyDown={handleKeyDown}
				role="dialog"
				aria-modal="true"
				aria-labelledby={titleId}
			>
				{/* biome-ignore lint/a11y/noStaticElementInteractions: modal content wrapper */}
				<div
					className="filter-modal"
					onClick={(e) => e.stopPropagation()}
					onKeyDown={(e) => e.stopPropagation()}
				>
					<div className="filter-modal-header">
						<div className="filter-modal-icon">
							<FilterIcon />
						</div>
						<h2 id={titleId} className="filter-modal-title">
							Filter Documents
						</h2>
					</div>
					<p className="filter-modal-description">
						Select documents to restrict searches. When active, only selected
						documents will be searched.
					</p>
					<input
						type="text"
						className="filter-search"
						placeholder="Search documents..."
						value={searchTerm}
						onChange={(e) => setSearchTerm(e.target.value)}
					/>
					<div className="filter-list">
						{loading ? (
							<div className="filter-loading">Loading documents...</div>
						) : filteredDocuments.length === 0 ? (
							<div className="filter-empty">
								{searchTerm ? "No matching documents" : "No documents found"}
							</div>
						) : (
							filteredDocuments.map((doc) => {
								const displayName = getDisplayName(doc);
								return (
									<label key={doc.id} className="filter-item">
										<input
											type="checkbox"
											checked={localSelected.has(displayName)}
											onChange={() => toggleDocument(displayName)}
										/>
										<span className="filter-item-label">{displayName}</span>
									</label>
								);
							})
						)}
					</div>
					<div className="filter-footer">
						<div className="filter-count">
							{localSelected.size > 0 ? (
								<>
									<strong>{localSelected.size}</strong> document
									{localSelected.size === 1 ? "" : "s"} selected
									<button
										type="button"
										className="filter-btn filter-btn-clear"
										onClick={handleClearAll}
									>
										Clear all
									</button>
								</>
							) : (
								"No filter (all documents)"
							)}
						</div>
						<div className="filter-buttons">
							<button
								type="button"
								className="filter-btn filter-btn-secondary"
								onClick={onClose}
							>
								Cancel
							</button>
							<button
								type="button"
								className="filter-btn filter-btn-primary"
								onClick={handleApply}
							>
								Apply
							</button>
						</div>
					</div>
				</div>
			</div>
		</>
	);
}
