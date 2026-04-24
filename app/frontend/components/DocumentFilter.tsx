"use client";

import { useCallback, useEffect, useId, useState } from "react";
import { FilterIcon } from "../lib/icons";

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

const getDisplayName = (doc: Document) => doc.title || doc.uri || doc.id;

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
	// Track selection by document id — two docs can share a title, but ids
	// are unique. Display names are only used for rendering and for the
	// filter string returned to the parent.
	const [localSelected, setLocalSelected] = useState<Set<string>>(new Set());

	// Refetch on every open so newly-added or deleted documents show up.
	useEffect(() => {
		if (!isOpen) return;
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
	}, [isOpen]);

	// Seed local selection from the parent's display-name list once documents
	// are available. Any doc whose display name is in `selected` starts checked.
	useEffect(() => {
		if (!isOpen) return;
		const selectedNames = new Set(selected);
		setLocalSelected(
			new Set(
				documents
					.filter((d) => selectedNames.has(getDisplayName(d)))
					.map((d) => d.id),
			),
		);
		setSearchTerm("");
	}, [isOpen, selected, documents]);

	const handleKeyDown = useCallback(
		(e: React.KeyboardEvent) => {
			if (e.key === "Escape") {
				onClose();
			}
		},
		[onClose],
	);

	const toggleDocument = (docId: string) => {
		setLocalSelected((prev) => {
			const next = new Set(prev);
			if (next.has(docId)) {
				next.delete(docId);
			} else {
				next.add(docId);
			}
			return next;
		});
	};

	const handleApply = () => {
		const names = documents
			.filter((d) => localSelected.has(d.id))
			.map(getDisplayName);
		// Dedupe: two selected docs sharing a title collapse to one filter term.
		onApply(Array.from(new Set(names)));
		onClose();
	};

	const handleClearAll = () => {
		setLocalSelected(new Set());
	};

	const filteredDocuments = documents.filter((doc) => {
		if (!searchTerm) return true;
		const displayName = getDisplayName(doc).toLowerCase();
		return displayName.includes(searchTerm.toLowerCase());
	});

	if (!isOpen) {
		return null;
	}

	return (
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
						<FilterIcon size={24} strokeWidth={1.5} />
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
										checked={localSelected.has(doc.id)}
										onChange={() => toggleDocument(doc.id)}
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
	);
}
