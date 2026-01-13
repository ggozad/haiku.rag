"use client";

import { useCallback, useState } from "react";

interface Citation {
	index: number;
	document_id: string;
	chunk_id: string;
	document_uri: string;
	document_title: string | null;
	page_numbers: number[];
	headings: string[] | null;
	content: string;
}

interface CitationBlockProps {
	citations: Citation[];
}

interface VisualGroundingState {
	isOpen: boolean;
	chunkId: string | null;
	images: string[];
	loading: boolean;
	error: string | null;
}

function CitationItem({
	citation,
	onViewInDocument,
}: {
	citation: Citation;
	onViewInDocument: (chunkId: string) => void;
}) {
	const [expanded, setExpanded] = useState(false);

	const title = citation.document_title || citation.document_uri || "Unknown";
	const pageInfo =
		citation.page_numbers.length > 0
			? `p. ${citation.page_numbers.join(", ")}`
			: null;

	return (
		<div className="citation-item">
			<button
				type="button"
				className="citation-header"
				onClick={() => setExpanded(!expanded)}
			>
				<span className="citation-index">[{citation.index}]</span>
				<span className="citation-title">{title}</span>
				{pageInfo && <span className="citation-page">{pageInfo}</span>}
				<span className={`citation-chevron ${expanded ? "expanded" : ""}`}>
					{expanded ? "▼" : "▶"}
				</span>
			</button>
			{expanded && (
				<div className="citation-content">
					{citation.headings && citation.headings.length > 0 && (
						<div className="citation-headings">
							{citation.headings.join(" › ")}
						</div>
					)}
					<div className="citation-text">{citation.content}</div>
					<button
						type="button"
						className="citation-view-btn"
						onClick={() => onViewInDocument(citation.chunk_id)}
					>
						View in Document
					</button>
				</div>
			)}
		</div>
	);
}

export default function CitationBlock({ citations }: CitationBlockProps) {
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
			const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "";
			const response = await fetch(`${backendUrl}/api/visualize/${chunkId}`);
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

	if (!citations || citations.length === 0) {
		return null;
	}

	return (
		<>
			<style>{`
				.citation-block {
					margin-top: 0.75rem;
					border: 1px solid #e2e8f0;
					border-radius: 8px;
					overflow: hidden;
					font-size: 0.875rem;
				}
				.citation-block-header {
					background: #f8fafc;
					padding: 0.5rem 0.75rem;
					font-weight: 500;
					color: #475569;
					border-bottom: 1px solid #e2e8f0;
				}
				.citation-item {
					border-bottom: 1px solid #f1f5f9;
				}
				.citation-item:last-child {
					border-bottom: none;
				}
				.citation-header {
					display: flex;
					align-items: center;
					gap: 0.5rem;
					width: 100%;
					padding: 0.5rem 0.75rem;
					background: none;
					border: none;
					cursor: pointer;
					text-align: left;
					transition: background 0.15s;
				}
				.citation-header:hover {
					background: #f8fafc;
				}
				.citation-index {
					font-weight: 600;
					color: #3b82f6;
					flex-shrink: 0;
				}
				.citation-title {
					flex: 1;
					color: #334155;
					overflow: hidden;
					text-overflow: ellipsis;
					white-space: nowrap;
				}
				.citation-page {
					color: #94a3b8;
					font-size: 0.75rem;
					flex-shrink: 0;
				}
				.citation-chevron {
					color: #94a3b8;
					font-size: 0.625rem;
					flex-shrink: 0;
					transition: transform 0.15s;
				}
				.citation-chevron.expanded {
					transform: rotate(0deg);
				}
				.citation-content {
					padding: 0.75rem;
					background: #fafafa;
					border-top: 1px solid #f1f5f9;
				}
				.citation-headings {
					font-size: 0.75rem;
					color: #64748b;
					margin-bottom: 0.5rem;
					font-style: italic;
				}
				.citation-text {
					color: #475569;
					line-height: 1.5;
					white-space: pre-wrap;
					max-height: 200px;
					overflow-y: auto;
				}
				.citation-view-btn {
					margin-top: 0.5rem;
					padding: 0.25rem 0.5rem;
					font-size: 0.75rem;
					background: #3b82f6;
					color: white;
					border: none;
					border-radius: 4px;
					cursor: pointer;
					transition: background 0.15s;
				}
				.citation-view-btn:hover {
					background: #2563eb;
				}
				.visual-modal-overlay {
					position: fixed;
					top: 0;
					left: 0;
					right: 0;
					bottom: 0;
					background: rgba(0, 0, 0, 0.75);
					display: flex;
					align-items: center;
					justify-content: center;
					z-index: 1000;
				}
				.visual-modal {
					background: white;
					border-radius: 8px;
					padding: 1.5rem;
					max-width: 90vw;
					max-height: 90vh;
					overflow: auto;
					position: relative;
				}
				.visual-modal-close {
					position: absolute;
					top: 0.5rem;
					right: 0.5rem;
					background: #ef4444;
					color: white;
					border: none;
					border-radius: 50%;
					width: 2rem;
					height: 2rem;
					cursor: pointer;
					font-size: 1rem;
					display: flex;
					align-items: center;
					justify-content: center;
				}
				.visual-modal-close:hover {
					background: #dc2626;
				}
				.visual-modal-title {
					margin: 0 0 1rem 0;
					font-size: 1.125rem;
					color: #1e293b;
				}
				.visual-modal-loading {
					padding: 2rem;
					text-align: center;
					color: #64748b;
				}
				.visual-modal-error {
					padding: 1rem;
					background: #fef2f2;
					color: #dc2626;
					border-radius: 4px;
				}
				.visual-modal-images {
					display: flex;
					flex-direction: column;
					gap: 1rem;
				}
				.visual-modal-page-label {
					font-size: 0.75rem;
					color: #64748b;
					margin-bottom: 0.5rem;
				}
				.visual-modal-image {
					max-width: 100%;
					border: 1px solid #e2e8f0;
					border-radius: 4px;
				}
			`}</style>
			<div className="citation-block">
				<div className="citation-block-header">
					Sources ({citations.length})
				</div>
				{citations.map((citation) => (
					<CitationItem
						key={citation.chunk_id}
						citation={citation}
						onViewInDocument={fetchVisualGrounding}
					/>
				))}
			</div>

			{visualGrounding.isOpen && (
				<div
					className="visual-modal-overlay"
					onClick={closeVisualGrounding}
					onKeyDown={(e) => e.key === "Escape" && closeVisualGrounding()}
					role="dialog"
					aria-modal="true"
					aria-label="Visual grounding"
				>
					{/* biome-ignore lint/a11y/noStaticElementInteractions: modal content wrapper */}
					<div
						className="visual-modal"
						onClick={(e) => e.stopPropagation()}
						onKeyDown={(e) => e.stopPropagation()}
					>
						<button
							type="button"
							className="visual-modal-close"
							onClick={closeVisualGrounding}
						>
							✕
						</button>
						<h3 className="visual-modal-title">Visual Grounding</h3>
						{visualGrounding.loading && (
							<div className="visual-modal-loading">Loading...</div>
						)}
						{visualGrounding.error && (
							<div className="visual-modal-error">{visualGrounding.error}</div>
						)}
						{!visualGrounding.loading &&
							!visualGrounding.error &&
							visualGrounding.images.length > 0 && (
								<div className="visual-modal-images">
									{visualGrounding.images.map((img, idx) => (
										// biome-ignore lint/suspicious/noArrayIndexKey: images have no stable id
										<div key={idx}>
											<div className="visual-modal-page-label">
												Page {idx + 1} of {visualGrounding.images.length}
											</div>
											{/* biome-ignore lint/performance/noImgElement: base64 data URLs require img element */}
											<img
												src={`data:image/png;base64,${img}`}
												alt={`Page ${idx + 1}`}
												className="visual-modal-image"
											/>
										</div>
									))}
								</div>
							)}
					</div>
				</div>
			)}
		</>
	);
}
