"use client";

import { useEffect, useState } from "react";

interface DbInfoData {
	exists: boolean;
	path: string;
	documents: number;
	chunks: number;
	documents_bytes: number;
	chunks_bytes: number;
	has_vector_index: boolean;
}

function formatBytes(bytes: number): string {
	if (bytes === 0) return "0 B";
	const k = 1024;
	const sizes = ["B", "KB", "MB", "GB"];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return `${Number.parseFloat((bytes / k ** i).toFixed(1))} ${sizes[i]}`;
}

export default function DbInfo() {
	const [info, setInfo] = useState<DbInfoData | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "";
		fetch(`${backendUrl}/api/info`)
			.then((res) => res.json())
			.then(setInfo)
			.catch((err) => setError(err.message));
	}, []);

	if (error) {
		return (
			<div className="db-info db-info-error">
				<span>Database unavailable</span>
			</div>
		);
	}

	if (!info) {
		return (
			<div className="db-info db-info-loading">
				<span>Loading...</span>
			</div>
		);
	}

	if (!info.exists) {
		return (
			<div className="db-info db-info-empty">
				<span>No database found</span>
			</div>
		);
	}

	return (
		<>
			<style>{`
				.db-info {
					display: flex;
					gap: 1.5rem;
					padding: 0.5rem 1rem;
					font-size: 0.75rem;
					color: #64748b;
					background: #f8fafc;
					border-top: 1px solid #e2e8f0;
					justify-content: center;
				}
				.db-info-error {
					color: #dc2626;
					background: #fef2f2;
				}
				.db-info-loading {
					color: #64748b;
				}
				.db-info-empty {
					color: #f59e0b;
					background: #fffbeb;
				}
				.db-stat {
					display: flex;
					align-items: center;
					gap: 0.375rem;
				}
				.db-stat-value {
					font-weight: 600;
					color: #334155;
				}
				.db-stat-label {
					color: #94a3b8;
				}
				.db-index-badge {
					padding: 0.125rem 0.375rem;
					border-radius: 9999px;
					font-size: 0.625rem;
					font-weight: 500;
				}
				.db-index-badge.indexed {
					background: #dcfce7;
					color: #166534;
				}
				.db-index-badge.not-indexed {
					background: #fef3c7;
					color: #92400e;
				}
			`}</style>
			<div className="db-info">
				<div className="db-stat">
					<span className="db-stat-value">{info.documents}</span>
					<span className="db-stat-label">documents</span>
				</div>
				<div className="db-stat">
					<span className="db-stat-value">{info.chunks}</span>
					<span className="db-stat-label">chunks</span>
				</div>
				<div className="db-stat">
					<span className="db-stat-value">
						{formatBytes(info.documents_bytes + info.chunks_bytes)}
					</span>
					<span className="db-stat-label">total</span>
				</div>
				<div className="db-stat">
					<span
						className={`db-index-badge ${info.has_vector_index ? "indexed" : "not-indexed"}`}
					>
						{info.has_vector_index ? "indexed" : "no index"}
					</span>
				</div>
			</div>
		</>
	);
}
