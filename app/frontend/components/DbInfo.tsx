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
	);
}
