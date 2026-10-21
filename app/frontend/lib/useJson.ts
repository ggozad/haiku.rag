"use client";

import { useEffect, useRef, useState } from "react";
import { decoded } from "./http";

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
		const body = await decoded<T>(res);
		if (!signal.aborted) on.data(body);
	} catch (err) {
		if (!signal.aborted)
			on.error(err instanceof Error ? err.message : String(err));
	}
	if (!signal.aborted) on.done();
}

// GETs `url` whenever it changes, aborting the request it replaces, so a
// response for an earlier selection never lands. A null url reads nothing.
// A change of `refresh` reads the same url again and shows the last answer
// until the new one lands.
export function useJson<T>(
	url: string | null,
	refresh?: unknown,
): {
	data: T | null;
	error: string | null;
	loading: boolean;
} {
	const [data, setData] = useState<T | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const read = useRef(url);

	// biome-ignore lint/correctness/useExhaustiveDependencies: refresh only triggers a read
	useEffect(() => {
		if (read.current !== url) {
			read.current = url;
			setData(null);
			setError(null);
		}
		if (url === null) {
			setLoading(false);
			return;
		}
		const controller = new AbortController();
		setLoading(true);
		void load<T>(url, controller.signal, {
			data: (value) => {
				setData(value);
				setError(null);
			},
			error: setError,
			done: () => setLoading(false),
		});
		return () => controller.abort();
	}, [url, refresh]);

	return { data, error, loading };
}
