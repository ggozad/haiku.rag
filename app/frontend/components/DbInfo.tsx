"use client";

import { useEffect, useState } from "react";

interface DatabaseEntry {
	name: string;
	available: boolean;
	error: string | null;
	documents?: number;
	chunks?: number;
}

export default function DbInfo() {
	const [databases, setDatabases] = useState<DatabaseEntry[] | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		fetch("/api/backend/api/databases")
			.then(async (res) => {
				const data = await res.json();
				if (!res.ok) throw new Error(data.error || res.statusText);
				setDatabases(data.databases ?? []);
			})
			.catch((err: Error) => setError(err.message));
	}, []);

	if (error) {
		return (
			<div className="db-info db-info-error" title={error}>
				<span>Backend unavailable</span>
			</div>
		);
	}

	if (!databases) {
		return (
			<div className="db-info db-info-loading">
				<span>Loading...</span>
			</div>
		);
	}

	return (
		<div className="db-info">
			{databases.map((db) =>
				db.available ? (
					<div key={db.name} className="db-stat">
						<span className="db-stat-label">{db.name}</span>
						<span className="db-stat-value">{db.documents}</span>
						<span className="db-stat-label">documents</span>
						<span className="db-stat-value">{db.chunks}</span>
						<span className="db-stat-label">chunks</span>
					</div>
				) : (
					<div
						key={db.name}
						className="db-stat db-info-error"
						title={db.error ?? undefined}
					>
						<span className="db-stat-label">{db.name}</span>
						<span className="db-index-badge not-indexed">unavailable</span>
					</div>
				),
			)}
		</div>
	);
}
