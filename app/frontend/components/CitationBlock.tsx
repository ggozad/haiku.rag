"use client";

import { useState } from "react";

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

function CitationItem({ citation }: { citation: Citation }) {
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
				</div>
			)}
		</div>
	);
}

export default function CitationBlock({ citations }: CitationBlockProps) {
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
			`}</style>
			<div className="citation-block">
				<div className="citation-block-header">
					Sources ({citations.length})
				</div>
				{citations.map((citation) => (
					<CitationItem key={citation.chunk_id} citation={citation} />
				))}
			</div>
		</>
	);
}
