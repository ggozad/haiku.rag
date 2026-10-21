export type Severity = "ok" | "warn" | "fail";

export interface Check {
	name: string;
	severity: Severity;
	message: string;
	remediation: string | null;
	details: string[];
}

export interface Job<T> {
	kind: string;
	status: "queued" | "running" | "done" | "failed";
	progress: string[];
	queued_at: string;
	started_at: string | null;
	finished_at: string | null;
	result: T | null;
	error: string | null;
}

export function counts(checks: Check[]): Record<Severity, number> {
	const tally: Record<Severity, number> = { ok: 0, warn: 0, fail: 0 };
	for (const check of checks) tally[check.severity]++;
	return tally;
}

// Failures first, then warnings, each in the doctor's own order.
export function ordered(checks: Check[]): Check[] {
	const rank: Record<Severity, number> = { fail: 0, warn: 1, ok: 2 };
	return checks
		.map((check, index) => ({ check, index }))
		.sort(
			(a, b) =>
				rank[a.check.severity] - rank[b.check.severity] || a.index - b.index,
		)
		.map(({ check }) => check);
}

// The least memory a scan of the chunk vectors holds: one float32 per
// dimension of every chunk. Centroids and clustering add to it.
export function scanBytes(chunks: number, dimension: number): number {
	return chunks * dimension * 4;
}

export function formatBytes(n: number): string {
	if (n < 1024) return `${n} B`;
	const units = ["KB", "MB", "GB", "TB"];
	let value = n / 1024;
	let unit = 0;
	while (value >= 1024 && unit < units.length - 1) {
		value /= 1024;
		unit++;
	}
	return `${value.toFixed(1)} ${units[unit]}`;
}

export function settled(job: Job<unknown> | null): boolean {
	return job === null || job.status === "done" || job.status === "failed";
}
