"use client";

import { useEffect, useState } from "react";

export interface DatabaseEntry {
	name: string;
	available: boolean;
	error: string | null;
	documents?: number;
	chunks?: number;
	supports_image_search?: boolean;
}

export interface DatabasesInfo {
	vision: boolean;
	databases: DatabaseEntry[];
}

let pending: Promise<DatabasesInfo> | null = null;

// The configured databases, fetched once per page load and shared.
export function fetchDatabases(): Promise<DatabasesInfo> {
	pending ??= fetch("/api/backend/api/databases")
		.then(async (res) => {
			const data = await res.json();
			if (!res.ok) throw new Error(data.error || res.statusText);
			return data as DatabasesInfo;
		})
		.catch((error) => {
			// A failure is not kept: the next caller asks again.
			pending = null;
			throw error;
		});
	return pending;
}

export function useDatabases(): {
	info: DatabasesInfo | null;
	error: string | null;
} {
	const [info, setInfo] = useState<DatabasesInfo | null>(null);
	const [error, setError] = useState<string | null>(null);
	useEffect(() => {
		fetchDatabases()
			.then(setInfo)
			.catch((err: Error) => setError(err.message));
	}, []);
	return { info, error };
}
