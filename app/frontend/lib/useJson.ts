"use client";

import { useEffect, useState } from "react";

// GETs `url` and reports through `on` unless `signal` aborted first, so a
// request for a replaced selection never lands.
export async function load<T>(
	url: string,
	signal: AbortSignal,
	on: {
		data: (value: T) => void;
		error: (message: string) => void;
		done: () => void;
	},
): Promise<void> {
	try {
		const res = await fetch(url, { signal });
		const body = await res.json();
		if (!res.ok) throw new Error(body.error || res.statusText);
		if (!signal.aborted) on.data(body as T);
	} catch (err) {
		if (!signal.aborted)
			on.error(err instanceof Error ? err.message : String(err));
	}
	if (!signal.aborted) on.done();
}

// GETs `url` whenever it changes, aborting the request it replaces, so a
// response for an earlier selection never lands. A null url reads nothing.
export function useJson<T>(url: string | null): {
	data: T | null;
	error: string | null;
	loading: boolean;
} {
	const [data, setData] = useState<T | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	useEffect(() => {
		setData(null);
		setError(null);
		if (url === null) {
			setLoading(false);
			return;
		}
		const controller = new AbortController();
		setLoading(true);
		void load<T>(url, controller.signal, {
			data: setData,
			error: setError,
			done: () => setLoading(false),
		});
		return () => controller.abort();
	}, [url]);

	return { data, error, loading };
}
