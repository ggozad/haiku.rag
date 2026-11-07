"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Compared } from "@/lib/doctor";
import { withBasePath } from "@/lib/paths";
import { useJob } from "./useJob";

export interface Family {
	members: string[];
	keep: string;
	similarity: Record<string, number>;
	label: Record<string, string>;
}

export interface SimilarityResult {
	documents: Compared[];
	families: Family[];
	left_off: number;
	generation: string;
	neighbours: boolean;
}

export const doctorBase = (source: string) =>
	withBasePath(`/api/backend/api/doctor/${encodeURIComponent(source)}`);

// The database's latest similarity job. A finished job is read again every
// 30 s, so staleness and eviction show.
export function useSimilarityJob(source: string) {
	const base = doctorBase(source);
	const { job, read, error, start } = useJob<SimilarityResult>(
		`${base}/similarity`,
		"similarity",
		30_000,
	);
	const result = job?.status === "done" ? job.result : null;
	return {
		job,
		read,
		error,
		result,
		stale: (job as { stale?: boolean | null } | null)?.stale,
		busy: job?.status === "queued" || job?.status === "running",
		computedAt: result ? job?.finished_at : null,
		compute: () => start(`${base}/similarity`),
	};
}

export type SimilarityJob = ReturnType<typeof useSimilarityJob>;

// The button computing similarity, the job's progress and failure, and
// whether it is out of date.
export function SimilarityControls({
	similarity,
	ready,
}: {
	similarity: SimilarityJob;
	ready: boolean;
}) {
	const { job, error, result, stale, busy } = similarity;
	return (
		<div className="space-y-2">
			<div className="flex flex-wrap items-center gap-3">
				<Button
					size="sm"
					disabled={busy || !ready}
					onClick={similarity.compute}
				>
					{busy
						? "Computing…"
						: result
							? "Compute again"
							: "Compute similarity"}
				</Button>
				{stale && (
					<Badge variant="outline">Out of date: the database changed</Badge>
				)}
				{result && stale === null && (
					<Badge variant="outline">
						Whether the database changed is unknown: it cannot be opened
					</Badge>
				)}
				{result && (
					<span className="text-xs text-muted-foreground">
						{result.documents.length} documents compared
						{result.left_off > 0 &&
							` · ${result.left_off} without a usable embedding left out`}
					</span>
				)}
			</div>
			{error && <p className="text-sm text-destructive">{error}</p>}
			{busy && job && (
				<p className="text-sm text-muted-foreground">
					{job.status === "queued"
						? "Waiting for another job to finish…"
						: (job.progress.at(-1) ?? "Starting…")}
				</p>
			)}
			{job?.status === "failed" && (
				<p className="text-sm text-destructive">
					Computing similarity failed: {job.error}
				</p>
			)}
		</div>
	);
}

// The near-duplicate families, each linking to its kept document, and their
// YAML export.
export function FamilyList({
	source,
	families,
	hrefOf,
}: {
	source: string;
	families: Family[];
	hrefOf: (family: Family) => string;
}) {
	return (
		<div className="space-y-1">
			<div className="flex items-baseline gap-2">
				<h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
					Near-duplicate families ({families.length})
				</h3>
				<a
					href={`${doctorBase(source)}/duplicates.yaml`}
					className="ml-auto text-xs underline"
				>
					Download as YAML
				</a>
			</div>
			{families.length === 0 && <p className="text-muted-foreground">None.</p>}
			{families.map((family) => (
				<Link
					key={family.keep}
					href={hrefOf(family)}
					className="flex w-full justify-between gap-2 py-0.5 hover:underline"
				>
					<span className="truncate">
						{family.members.length} documents, keep {family.label[family.keep]}
					</span>
					<span className="shrink-0 font-mono text-xs tabular-nums">
						{Object.values(family.similarity)
							.reduce((a, b) => Math.min(a, b), 1)
							.toFixed(3)}
					</span>
				</Link>
			))}
		</div>
	);
}
