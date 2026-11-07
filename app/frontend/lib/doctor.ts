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

const count = (n: number, one: string, many: string) =>
	`${n} ${n === 1 ? one : many}`;

// The report in one line, toned by its worst result.
export function verdict(tally: Record<Severity, number>): {
	tone: Severity;
	label: string;
} {
	const warnings = tally.warn ? count(tally.warn, "warning", "warnings") : "";
	if (tally.fail) {
		const failed = `${count(tally.fail, "check", "checks")} failed`;
		return {
			tone: "fail",
			label: warnings ? `${failed}, ${warnings}` : failed,
		};
	}
	if (tally.warn) return { tone: "warn", label: warnings };
	return { tone: "ok", label: `All ${tally.ok} checks pass` };
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

export interface Compared {
	id: string;
	title: string | null;
	uri: string | null;
}

// The `k` documents most similar to `selected`, most similar first. Score `i`
// belongs to `documents[i]`, and scores from another job are ignored.
export function nearestDocuments(
	scores: { generation: string; similarity: number[] } | null,
	generation: string,
	documents: Compared[],
	selected: string,
	k: number,
): (Compared & { similarity: number })[] | null {
	if (!scores || scores.generation !== generation) return null;
	if (!documents.some((doc) => doc.id === selected)) return null;
	return documents
		.map((doc, index) => ({ ...doc, similarity: scores.similarity[index] }))
		.filter((doc) => doc.id !== selected)
		.sort((a, b) => b.similarity - a.similarity)
		.slice(0, k);
}
