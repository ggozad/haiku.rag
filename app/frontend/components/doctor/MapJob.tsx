"use client";

import Link from "next/link";
import type { Family } from "@/components/inspect/MapCanvas";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatBytes, scanBytes } from "@/lib/doctor";
import type { Point } from "@/lib/map";
import { useJson } from "@/lib/useJson";
import { useJob } from "./useJob";

export interface MapResult {
	points: Point[];
	families: Family[];
	left_off: number;
	generation: string;
	neighbours: boolean;
}

export const doctorBase = (source: string) =>
	`/api/backend/api/doctor/${encodeURIComponent(source)}`;

// The database's latest corpus map job. A finished map is read again every
// 30 s, so staleness and eviction show.
export function useMapJob(source: string) {
	const base = doctorBase(source);
	const { job, read, error, start } = useJob<MapResult>(
		`${base}/projection`,
		"projection",
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
		compute: () => start(`${base}/projection`),
	};
}

export type MapJob = ReturnType<typeof useMapJob>;

// The memory reading every chunk vector needs. `ready` once the figure, or why
// it is unknown, is shown, so the operator can decide before starting a job.
export function useEstimate(source: string | null) {
	const details = useJson<{
		info: {
			embeddings: { vector_dim: number | null };
			tables: { name: string; num_rows: number }[];
		};
	}>(
		source === null
			? null
			: `/api/backend/api/inspect/${encodeURIComponent(source)}/info`,
	);
	const dimension = details.data?.info.embeddings.vector_dim;
	const chunks = details.data?.info.tables.find(
		(table) => table.name === "chunks",
	)?.num_rows;
	const text =
		chunks !== undefined && dimension
			? `Reading the vectors holds at least ${formatBytes(scanBytes(chunks, dimension))} in memory.`
			: details.error
				? `The memory needed is unknown: ${details.error}`
				: null;
	return { ready: details.data !== null || details.error !== null, text };
}

export function Estimate({ text }: { text: string | null }) {
	return text ? (
		<span className="text-sm text-muted-foreground">{text}</span>
	) : null;
}

// Computing the map, its progress and failure, and whether it is out of date.
export function MapControls({ map, ready }: { map: MapJob; ready: boolean }) {
	const { job, error, result, stale, busy } = map;
	return (
		<div className="space-y-2">
			<div className="flex flex-wrap items-center gap-3">
				<Button size="sm" disabled={busy || !ready} onClick={map.compute}>
					{busy ? "Computing…" : result ? "Compute again" : "Compute the map"}
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
						{result.points.length} documents
						{result.left_off > 0 &&
							` · ${result.left_off} without a usable embedding left off`}
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
				<p className="text-sm text-destructive">The map failed: {job.error}</p>
			)}
		</div>
	);
}

// The map's near-duplicate families and their YAML export. A family either
// picks itself on the map beside it, or links to the map.
export function FamilyList({
	source,
	families,
	active = null,
	onPick,
	hrefOf,
}: {
	source: string;
	families: Family[];
	active?: number | null;
	onPick?: (index: number) => void;
	hrefOf?: (family: Family) => string;
}) {
	const row =
		"flex w-full justify-between gap-2 py-0.5 text-left hover:underline aria-pressed:font-medium";
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
			{families.map((family, index) => {
				const body = (
					<>
						<span className="truncate">
							{family.members.length} documents, keep{" "}
							{family.label[family.keep]}
						</span>
						<span className="shrink-0 font-mono text-xs tabular-nums">
							{Object.values(family.similarity)
								.reduce((a, b) => Math.min(a, b), 1)
								.toFixed(3)}
						</span>
					</>
				);
				return hrefOf ? (
					<Link key={family.keep} href={hrefOf(family)} className={row}>
						{body}
					</Link>
				) : (
					<button
						key={family.keep}
						type="button"
						aria-pressed={active === index}
						className={row}
						onClick={() => onPick?.(index)}
					>
						{body}
					</button>
				);
			})}
		</div>
	);
}
