"use client";

import { useCallback, useEffect, useState } from "react";
import { type Job, settled } from "@/lib/doctor";

// The latest job of one kind on one database, polled every second while it is
// queued or running. `start` posts and polls until the job settles.
export function useJob<T>(
	statusUrl: string | null,
	key: "run" | "projection",
	// After the job settles, read it again at this interval.
	revalidateMs?: number,
): {
	job: Job<T> | null;
	error: string | null;
	start: (url: string) => Promise<void>;
} {
	const [job, setJob] = useState<Job<T> | null>(null);
	const [error, setError] = useState<string | null>(null);
	// Bumped by `start`, so polling restarts from the posted job.
	const [generation, setGeneration] = useState(0);

	// One poll loop per status url and start: it keeps polling through failed
	// reads, stops once the job settles, and a replaced loop lands nothing.
	// biome-ignore lint/correctness/useExhaustiveDependencies: generation restarts the loop
	useEffect(() => {
		if (statusUrl === null) {
			setJob(null);
			return;
		}
		const controller = new AbortController();
		let timer: ReturnType<typeof setTimeout> | undefined;
		const poll = async () => {
			try {
				const res = await fetch(statusUrl, { signal: controller.signal });
				const body = await res.json();
				if (!res.ok) throw new Error(body.error || res.statusText);
				if (controller.signal.aborted) return;
				setJob(body[key]);
				setError(null);
				if (settled(body[key])) {
					if (revalidateMs !== undefined)
						timer = setTimeout(poll, revalidateMs);
					return;
				}
			} catch (err) {
				if (controller.signal.aborted) return;
				setError(err instanceof Error ? err.message : String(err));
			}
			timer = setTimeout(poll, 1000);
		};
		void poll();
		return () => {
			controller.abort();
			clearTimeout(timer);
		};
	}, [statusUrl, key, generation, revalidateMs]);

	const start = useCallback(async (url: string) => {
		const res = await fetch(url, { method: "POST" });
		const body = await res.json();
		if (!res.ok) {
			setError(body.error || res.statusText);
			return;
		}
		setError(null);
		setJob(body);
		setGeneration((n) => n + 1);
	}, []);

	return { job, error, start };
}
