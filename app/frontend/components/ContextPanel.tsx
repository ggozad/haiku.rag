"use client";

import { useCallback, useEffect, useId, useState } from "react";
import { formatRelativeTime } from "../lib/format";
import { BrainIcon } from "../lib/icons";
import type { SessionContext } from "../lib/sessionStorage";

interface ContextPanelProps {
	isOpen: boolean;
	onClose: () => void;
	sessionContext: SessionContext | null;
	initialContext?: string;
	onInitialContextChange?: (value: string) => void;
	isLocked?: boolean;
}

export default function ContextPanel({
	isOpen,
	onClose,
	sessionContext,
	initialContext = "",
	onInitialContextChange,
	isLocked = false,
}: ContextPanelProps) {
	const titleId = useId();
	const [localValue, setLocalValue] = useState(initialContext);

	useEffect(() => {
		if (isOpen) {
			setLocalValue(initialContext);
		}
	}, [isOpen, initialContext]);

	const handleKeyDown = useCallback(
		(e: React.KeyboardEvent) => {
			if (e.key === "Escape") {
				onClose();
			}
		},
		[onClose],
	);

	const handleSave = useCallback(() => {
		onInitialContextChange?.(localValue);
		onClose();
	}, [localValue, onInitialContextChange, onClose]);

	if (!isOpen) {
		return null;
	}

	const hasSessionContext = sessionContext?.summary?.trim();
	// Show edit mode when: not locked AND no session context yet
	const isEditMode = !isLocked && !hasSessionContext;

	return (
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
						<BrainIcon size={24} strokeWidth={1.5} />
					</div>
					<h2 id={titleId} className="context-modal-title">
						{isEditMode ? "Initial Context" : "Session Context"}
					</h2>
				</div>
				<p className="context-modal-description">
					{isEditMode
						? "Set background context to guide the conversation. This will be locked after you send your first message."
						: "This is what the assistant has learned from your conversation so far. It uses this context to provide more relevant answers."}
				</p>
				{isEditMode ? (
					<textarea
						className="context-textarea"
						placeholder="Enter any background context or instructions for the assistant..."
						value={localValue}
						onChange={(e) => setLocalValue(e.target.value)}
					/>
				) : hasSessionContext ? (
					<div className="context-content">{sessionContext.summary}</div>
				) : (
					<div className="context-empty">
						<div className="context-empty-icon">
							<BrainIcon size={24} strokeWidth={1.5} />
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
					<div className="context-footer-buttons">
						<button
							type="button"
							className="context-btn context-btn-close"
							onClick={onClose}
						>
							{isEditMode ? "Cancel" : "Close"}
						</button>
						{isEditMode && (
							<button
								type="button"
								className="context-btn context-btn-save"
								onClick={handleSave}
							>
								Save
							</button>
						)}
					</div>
				</div>
			</div>
		</div>
	);
}
