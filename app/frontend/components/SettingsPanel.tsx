"use client";

import { useCallback, useEffect, useId, useState } from "react";

const STORAGE_KEY = "haiku.rag.settings.background_context";

interface SettingsPanelProps {
	isOpen: boolean;
	onClose: () => void;
	onSave: (backgroundContext: string) => void;
	currentValue: string;
}

export default function SettingsPanel({
	isOpen,
	onClose,
	onSave,
	currentValue,
}: SettingsPanelProps) {
	const [value, setValue] = useState(currentValue);
	const titleId = useId();

	useEffect(() => {
		if (isOpen) {
			setValue(currentValue);
		}
	}, [isOpen, currentValue]);

	const handleSave = useCallback(() => {
		onSave(value);
		onClose();
	}, [value, onSave, onClose]);

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

	return (
		<>
			<style>{`
				.settings-modal-overlay {
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
				.settings-modal {
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
				.settings-modal-title {
					margin: 0 0 0.5rem 0;
					font-size: 1.25rem;
					font-weight: 600;
					color: #1e293b;
				}
				.settings-modal-description {
					margin: 0 0 1rem 0;
					font-size: 0.875rem;
					color: #64748b;
					line-height: 1.5;
				}
				.settings-textarea {
					width: 100%;
					min-height: 150px;
					padding: 0.75rem;
					border: 1px solid #e2e8f0;
					border-radius: 8px;
					font-size: 0.875rem;
					font-family: inherit;
					resize: vertical;
					color: #334155;
					line-height: 1.5;
				}
				.settings-textarea:focus {
					outline: none;
					border-color: #3b82f6;
					box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
				}
				.settings-textarea::placeholder {
					color: #94a3b8;
				}
				.settings-modal-actions {
					display: flex;
					justify-content: flex-end;
					gap: 0.75rem;
					margin-top: 1.25rem;
				}
				.settings-btn {
					padding: 0.5rem 1rem;
					font-size: 0.875rem;
					font-weight: 500;
					border-radius: 6px;
					cursor: pointer;
					transition: all 0.15s;
				}
				.settings-btn-cancel {
					background: white;
					color: #475569;
					border: 1px solid #e2e8f0;
				}
				.settings-btn-cancel:hover {
					background: #f8fafc;
					border-color: #cbd5e1;
				}
				.settings-btn-save {
					background: #3b82f6;
					color: white;
					border: none;
				}
				.settings-btn-save:hover {
					background: #2563eb;
				}
			`}</style>
			<div
				className="settings-modal-overlay"
				onClick={onClose}
				onKeyDown={handleKeyDown}
				role="dialog"
				aria-modal="true"
				aria-labelledby={titleId}
			>
				{/* biome-ignore lint/a11y/noStaticElementInteractions: modal content wrapper */}
				<div
					className="settings-modal"
					onClick={(e) => e.stopPropagation()}
					onKeyDown={(e) => e.stopPropagation()}
				>
					<h2 id={titleId} className="settings-modal-title">
						Background Context
					</h2>
					<p className="settings-modal-description">
						Provide background information that will be used throughout your
						conversation. This helps the assistant understand domain-specific
						context, terminology, or any relevant details about your questions.
					</p>
					<textarea
						className="settings-textarea"
						value={value}
						onChange={(e) => setValue(e.target.value)}
						placeholder="e.g., Focus on Python programming concepts and best practices..."
						autoFocus
					/>
					<div className="settings-modal-actions">
						<button
							type="button"
							className="settings-btn settings-btn-cancel"
							onClick={onClose}
						>
							Cancel
						</button>
						<button
							type="button"
							className="settings-btn settings-btn-save"
							onClick={handleSave}
						>
							Save
						</button>
					</div>
				</div>
			</div>
		</>
	);
}

export { STORAGE_KEY };
