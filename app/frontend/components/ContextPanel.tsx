"use client";

import { useCallback, useId } from "react";

interface SessionContext {
	summary: string;
	last_updated: string | null;
}

interface ContextPanelProps {
	isOpen: boolean;
	onClose: () => void;
	sessionContext: SessionContext | null;
}

function formatRelativeTime(isoString: string): string {
	const date = new Date(isoString);
	const now = new Date();
	const diffMs = now.getTime() - date.getTime();
	const diffSec = Math.floor(diffMs / 1000);
	const diffMin = Math.floor(diffSec / 60);
	const diffHour = Math.floor(diffMin / 60);

	if (diffSec < 60) {
		return "just now";
	}
	if (diffMin < 60) {
		return `${diffMin} minute${diffMin === 1 ? "" : "s"} ago`;
	}
	if (diffHour < 24) {
		return `${diffHour} hour${diffHour === 1 ? "" : "s"} ago`;
	}
	return date.toLocaleDateString();
}

function BrainIcon() {
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
			<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
			<path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
			<path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
			<path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
			<path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
			<path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
			<path d="M19.938 10.5a4 4 0 0 1 .585.396" />
			<path d="M6 18a4 4 0 0 1-1.967-.516" />
			<path d="M19.967 17.484A4 4 0 0 1 18 18" />
		</svg>
	);
}

export default function ContextPanel({
	isOpen,
	onClose,
	sessionContext,
}: ContextPanelProps) {
	const titleId = useId();

	const handleKeyDown = useCallback(
		(e: React.KeyboardEvent) => {
			if (e.key === "Escape") {
				onClose();
			}
		},
		[onClose],
	);

	if (!isOpen) {
		return null;
	}

	const hasContext = sessionContext?.summary && sessionContext.summary.trim();

	return (
		<>
			<style>{`
				.context-modal-overlay {
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
				.context-modal {
					background: white;
					border-radius: 12px;
					padding: 1.5rem;
					width: 90%;
					max-width: 600px;
					max-height: 80vh;
					overflow: auto;
					position: relative;
					box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
				}
				.context-modal-header {
					display: flex;
					align-items: center;
					gap: 0.75rem;
					margin-bottom: 0.5rem;
				}
				.context-modal-icon {
					display: flex;
					align-items: center;
					justify-content: center;
					width: 40px;
					height: 40px;
					border-radius: 10px;
					background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
					color: #3b82f6;
				}
				.context-modal-title {
					margin: 0;
					font-size: 1.25rem;
					font-weight: 600;
					color: #1e293b;
				}
				.context-modal-description {
					margin: 0 0 1rem 0;
					font-size: 0.875rem;
					color: #64748b;
					line-height: 1.5;
				}
				.context-content {
					background: #f8fafc;
					border: 1px solid #e2e8f0;
					border-radius: 8px;
					padding: 1rem;
					font-size: 0.875rem;
					line-height: 1.6;
					color: #334155;
					white-space: pre-wrap;
					max-height: 400px;
					overflow-y: auto;
				}
				.context-empty {
					display: flex;
					flex-direction: column;
					align-items: center;
					justify-content: center;
					padding: 2rem 1rem;
					text-align: center;
					color: #94a3b8;
				}
				.context-empty-icon {
					margin-bottom: 0.75rem;
					opacity: 0.5;
				}
				.context-empty-text {
					font-size: 0.875rem;
					line-height: 1.5;
				}
				.context-footer {
					display: flex;
					justify-content: space-between;
					align-items: center;
					margin-top: 1rem;
					padding-top: 1rem;
					border-top: 1px solid #e2e8f0;
				}
				.context-timestamp {
					font-size: 0.75rem;
					color: #94a3b8;
				}
				.context-btn-close {
					padding: 0.5rem 1rem;
					font-size: 0.875rem;
					font-weight: 500;
					border-radius: 6px;
					cursor: pointer;
					transition: all 0.15s;
					background: white;
					color: #475569;
					border: 1px solid #e2e8f0;
				}
				.context-btn-close:hover {
					background: #f8fafc;
					border-color: #cbd5e1;
				}
			`}</style>
			<div
				className="context-modal-overlay"
				onClick={onClose}
				onKeyDown={handleKeyDown}
				role="dialog"
				aria-modal="true"
				aria-labelledby={titleId}
			>
				{/* biome-ignore lint/a11y/noStaticElementInteractions: modal content wrapper */}
				<div
					className="context-modal"
					onClick={(e) => e.stopPropagation()}
					onKeyDown={(e) => e.stopPropagation()}
				>
					<div className="context-modal-header">
						<div className="context-modal-icon">
							<BrainIcon />
						</div>
						<h2 id={titleId} className="context-modal-title">
							Session Context
						</h2>
					</div>
					<p className="context-modal-description">
						This is what the assistant has learned from your conversation so
						far. It uses this context to provide more relevant answers.
					</p>
					{hasContext ? (
						<div className="context-content">{sessionContext.summary}</div>
					) : (
						<div className="context-empty">
							<div className="context-empty-icon">
								<BrainIcon />
							</div>
							<div className="context-empty-text">
								No context yet. Ask some questions to build context.
							</div>
						</div>
					)}
					<div className="context-footer">
						<span className="context-timestamp">
							{sessionContext?.last_updated
								? `Last updated: ${formatRelativeTime(sessionContext.last_updated)}`
								: ""}
						</span>
						<button
							type="button"
							className="context-btn-close"
							onClick={onClose}
						>
							Close
						</button>
					</div>
				</div>
			</div>
		</>
	);
}
