"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Citation } from "../lib/sessionStorage";

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
	// AbortController for the in-flight visualize fetch so a rapid close/reopen
	// doesn't let a stale response overwrite the new request's state.
	const abortRef = useRef<AbortController | null>(null);

	// Abort any in-flight request on unmount.
	useEffect(() => {
		return () => abortRef.current?.abort();
	}, []);

	const fetchVisualGrounding = useCallback(async (chunkId: string) => {
		abortRef.current?.abort();
		const controller = new AbortController();
		abortRef.current = controller;

		setVisualGrounding({
			isOpen: true,
			chunkId,
			images: [],
			loading: true,
			error: null,
		});

		try {
			const response = await fetch(`/api/visualize/${chunkId}`, {
				signal: controller.signal,
			});
			const data = await response.json();

			if (controller.signal.aborted) return;
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
			if (controller.signal.aborted) return;
			setVisualGrounding((prev) => ({
				...prev,
				loading: false,
				error: err instanceof Error ? err.message : "Unknown error",
			}));
		}
	}, []);

	const closeVisualGrounding = useCallback(() => {
		abortRef.current?.abort();
		abortRef.current = null;
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
