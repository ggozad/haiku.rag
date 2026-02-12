"use client";

import { useEffect, useRef, useState } from "react";
import { formatRelativeTime } from "../lib/format";
import {
	createSession,
	deleteSession,
	exportSessionToMarkdown,
	getAllSessions,
	type StoredSession,
	setActiveSessionId,
} from "../lib/sessionStorage";

interface SessionManagerProps {
	activeSessionId: string | null;
	onSessionChange: (sessionId: string) => void;
}

function HistoryIcon() {
	return (
		<svg
			width="18"
			height="18"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
			<path d="M3 3v5h5" />
			<path d="M12 7v5l4 2" />
		</svg>
	);
}

function PlusIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2.5"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M12 5v14" />
			<path d="M5 12h14" />
		</svg>
	);
}

function DownloadIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
			<polyline points="7 10 12 15 17 10" />
			<line x1="12" y1="15" x2="12" y2="3" />
		</svg>
	);
}

function TrashIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M3 6h18" />
			<path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
			<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
		</svg>
	);
}

export default function SessionManager({
	activeSessionId,
	onSessionChange,
}: SessionManagerProps) {
	const [isOpen, setIsOpen] = useState(false);
	const [sessions, setSessions] = useState<StoredSession[]>([]);
	const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
	const dropdownRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (isOpen) setSessions(getAllSessions());
	}, [isOpen]);

	useEffect(() => {
		function handleClickOutside(e: MouseEvent) {
			if (
				dropdownRef.current &&
				!dropdownRef.current.contains(e.target as Node)
			) {
				setIsOpen(false);
				setConfirmDelete(null);
			}
		}
		if (isOpen) document.addEventListener("mousedown", handleClickOutside);
		return () => document.removeEventListener("mousedown", handleClickOutside);
	}, [isOpen]);

	const handleNewSession = () => {
		const session = createSession();
		setSessions(getAllSessions());
		setIsOpen(false);
		onSessionChange(session.id);
	};

	const handleSelectSession = (id: string) => {
		setActiveSessionId(id);
		setIsOpen(false);
		onSessionChange(id);
	};

	const handleDelete = (id: string) => {
		deleteSession(id);
		const remaining = getAllSessions();
		setSessions(remaining);
		setConfirmDelete(null);
		if (id === activeSessionId) {
			if (remaining.length > 0) {
				setActiveSessionId(remaining[0].id);
				onSessionChange(remaining[0].id);
			} else {
				const session = createSession();
				setSessions(getAllSessions());
				onSessionChange(session.id);
			}
		}
	};

	const handleExport = (session: StoredSession) => {
		exportSessionToMarkdown(session);
	};

	const activeTitle =
		sessions.find((s) => s.id === activeSessionId)?.title ?? "Sessions";

	return (
		<div ref={dropdownRef} style={{ position: "relative" }}>
			<button
				type="button"
				className="header-btn"
				onClick={() => setIsOpen(!isOpen)}
				title="Session history"
			>
				<HistoryIcon />
				<span
					style={{
						maxWidth: 120,
						overflow: "hidden",
						textOverflow: "ellipsis",
						whiteSpace: "nowrap",
					}}
				>
					{activeTitle}
				</span>
			</button>

			{isOpen && (
				<div className="session-dropdown">
					<div className="session-dropdown-header">
						<span>Sessions</span>
						<button
							type="button"
							className="new-session-btn"
							onClick={handleNewSession}
						>
							<PlusIcon />
							New
						</button>
					</div>
					<div className="session-list">
						{sessions.length === 0 && (
							<div
								style={{
									padding: "16px",
									textAlign: "center",
									color: "#94a3b8",
									fontSize: "13px",
								}}
							>
								No sessions yet
							</div>
						)}
						{sessions.map((session) => (
							<div
								key={session.id}
								className={`session-item ${session.id === activeSessionId ? "active" : ""}`}
							>
								<button
									type="button"
									className="session-item-content"
									onClick={() => handleSelectSession(session.id)}
									onKeyDown={(e) => {
										if (e.key === "Enter") handleSelectSession(session.id);
									}}
								>
									<div className="session-item-title">{session.title}</div>
									<div className="session-item-meta">
										<span>{session.messages.length} messages</span>
										<span>{formatRelativeTime(session.updatedAt, true)}</span>
									</div>
								</button>
								{confirmDelete === session.id ? (
									<div className="confirm-delete">
										<button
											type="button"
											className="confirm-yes"
											onClick={() => handleDelete(session.id)}
										>
											Delete
										</button>
										<button
											type="button"
											className="confirm-no"
											onClick={() => setConfirmDelete(null)}
										>
											Cancel
										</button>
									</div>
								) : (
									<div className="session-actions">
										<button
											type="button"
											className="session-action-btn"
											onClick={() => handleExport(session)}
											title="Export to markdown"
										>
											<DownloadIcon />
										</button>
										<button
											type="button"
											className="session-action-btn danger"
											onClick={() => setConfirmDelete(session.id)}
											title="Delete session"
										>
											<TrashIcon />
										</button>
									</div>
								)}
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
}
