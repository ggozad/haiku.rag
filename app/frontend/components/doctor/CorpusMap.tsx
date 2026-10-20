"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { type ColourBy, categories, type Point, shading } from "@/lib/map";
import { useJson } from "@/lib/useJson";
import MapCanvas, { type Family, type Paint } from "./MapCanvas";
import { useJob } from "./useJob";

interface MapResult {
	points: Point[];
	families: Family[];
	left_off: number;
	generation: string;
	neighbours: boolean;
}

const SLOT_TOKENS = ["--chart-1", "--chart-2", "--chart-3"];
const OTHER = "--muted-foreground";

const nameOf = (point: Point) => point.title || point.uri || point.id;

export default function CorpusMap({
	source,
	ready,
}: {
	source: string;
	// The memory figure is shown, so the operator can decide.
	ready: boolean;
}) {
	const base = `/api/backend/api/doctor/${encodeURIComponent(source)}`;
	// A finished map is read again every 30 s, so staleness and eviction show.
	const { job, error, start } = useJob<MapResult>(
		`${base}/projection`,
		"projection",
		30_000,
	);
	const result = job?.status === "done" ? job.result : null;
	const stale = (job as { stale?: boolean | null } | null)?.stale;
	const points = result?.points ?? [];

	// Held by id, so a recomputed map resolves them against its own points.
	const [hoveredId, setHoveredId] = useState<string | null>(null);
	const [selectedId, setSelectedId] = useState<string | null>(null);
	const setHovered = (point: Point | null) => setHoveredId(point?.id ?? null);
	const setSelected = (point: Point | null) => setSelectedId(point?.id ?? null);
	const [family, setFamily] = useState<number | null>(null);
	const [colourBy, setColourBy] = useState<ColourBy>("family");
	const [term, setTerm] = useState("");

	const { slotOf, legend } = useMemo(
		() => categories(points, colourBy),
		[points, colourBy],
	);
	const matching = useMemo(() => {
		const needle = term.trim().toLowerCase();
		return needle
			? new Set(
					points
						.filter((p) => nameOf(p).toLowerCase().includes(needle))
						.map((p) => p.id),
				)
			: null;
	}, [points, term]);
	const byId = useMemo(() => new Map(points.map((p) => [p.id, p])), [points]);
	const selected = (selectedId && byId.get(selectedId)) || null;
	const hovered = (hoveredId && byId.get(hoveredId)) || null;
	const similar = useJson<{ generation: string; similarity: number[] }>(
		selected && result?.neighbours
			? `${base}/projection/similarity?${new URLSearchParams({ id: selected.id, generation: result.generation })}`
			: null,
	);
	// While a document is selected, every other is shaded by its closeness to it.
	const shade = useMemo(
		() => shading(similar.data, result?.generation, points, selectedId),
		[similar.data, result?.generation, points, selectedId],
	);

	// The colour token a point is filled with, and whether it has no category.
	const paint = useCallback(
		(point: Point): Paint => {
			const slot = slotOf(point);
			if (colourBy === "family")
				return {
					token: point.family === null ? OTHER : "--destructive",
					muted: slot === null,
				};
			return {
				token: slot !== null && slot < SLOTS_SHOWN ? SLOT_TOKENS[slot] : OTHER,
				muted: slot === null,
			};
		},
		[slotOf, colourBy],
	);

	// Drop the family selection and any selection not on the new map when a new
	// map arrives, not when the same map is read again.
	const computedAt = result ? job?.finished_at : null;
	// biome-ignore lint/correctness/useExhaustiveDependencies: a new map has a new finish time
	useEffect(() => {
		setFamily(null);
		if (!result) return;
		setSelectedId((id) => (id && byId.has(id) ? id : null));
		setHoveredId((id) => (id && byId.has(id) ? id : null));
	}, [computedAt]);

	const busy = job?.status === "queued" || job?.status === "running";

	return (
		<section className="space-y-3">
			<div className="flex flex-wrap items-center gap-3">
				<h2 className="font-semibold">Corpus map</h2>
				<Button
					size="sm"
					disabled={busy || !ready}
					onClick={() => start(`${base}/projection`)}
				>
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
						{points.length} documents
						{result.left_off > 0 &&
							` · ${result.left_off} without a usable embedding left off`}
					</span>
				)}
			</div>
			{error && <p className="text-sm text-destructive">{error}</p>}
			{busy && (
				<p className="text-sm text-muted-foreground">
					{job.status === "queued"
						? "Waiting for another job to finish…"
						: (job.progress.at(-1) ?? "Starting…")}
				</p>
			)}
			{job?.status === "failed" && (
				<p className="text-sm text-destructive">The map failed: {job.error}</p>
			)}
			{result && (
				<>
					<div className="flex flex-wrap items-center gap-3 text-sm">
						<Select
							value={colourBy}
							onValueChange={(value) => setColourBy(value as ColourBy)}
						>
							<SelectTrigger className="w-48" aria-label="Colour by">
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="family">Near-duplicate families</SelectItem>
								<SelectItem value="source_id">Source</SelectItem>
								<SelectItem value="content_type">Content type</SelectItem>
							</SelectContent>
						</Select>
						<Input
							id="map-search"
							className="w-64"
							placeholder="Find a document by title or URI"
							value={term}
							onChange={(event) => setTerm(event.target.value)}
						/>
						<div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
							{shade && (
								<span className="flex items-center gap-1">
									least similar
									<span
										className="inline-block h-2.5 w-24 rounded-full"
										style={{
											background:
												// The shading's cube: 12.5% at the median, 51% at the 80th percentile.
												"linear-gradient(to right, var(--map-far), color-mix(in srgb, var(--map-near) 12.5%, var(--map-far)) 50%, color-mix(in srgb, var(--map-near) 51%, var(--map-far)) 80%, var(--map-near))",
										}}
									/>
									most similar to the selected document · click empty space to
									clear
								</span>
							)}
							{!shade &&
								legend.map((label, slot) => (
									<span key={label} className="flex items-center gap-1">
										<span
											className="inline-block size-2.5 rounded-full"
											style={{
												background:
													colourBy === "family"
														? "var(--destructive)"
														: slot < SLOTS_SHOWN
															? `var(${SLOT_TOKENS[slot]})`
															: "var(--muted-foreground)",
											}}
										/>
										{label}
									</span>
								))}
						</div>
						<a
							href={`${base}/duplicates.yaml`}
							className="ml-auto text-xs underline"
						>
							Download the families as YAML
						</a>
					</div>
					<div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
						<MapCanvas
							points={points}
							families={result.families}
							family={family}
							paint={paint}
							matching={matching}
							selected={selected}
							hovered={hovered}
							shade={shade}
							fitKey={computedAt}
							label={(point) => `${nameOf(point)} · ${point.chunks} chunks`}
							onHover={setHovered}
							onSelect={setSelected}
						/>
						<aside className="min-w-0 space-y-4 text-sm">
							<div className="space-y-1">
								<h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
									Selected
								</h3>
								{selected ? (
									<>
										<p className="break-words font-medium">
											{nameOf(selected)}
										</p>
										<p className="text-xs text-muted-foreground">
											{selected.chunks} chunks
											{selected.family !== null &&
												` · family ${selected.family + 1}`}
										</p>
										<Link
											className="text-xs underline"
											href={`/inspect?${new URLSearchParams({ source, document: selected.id })}`}
										>
											Open in the inspector
										</Link>
									</>
								) : (
									<p className="text-muted-foreground">Click a point.</p>
								)}
							</div>
							{selected && (
								<div className="space-y-1">
									<h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
										Nearest documents
									</h3>
									{!result.neighbours && (
										<p className="text-muted-foreground">
											Unavailable: the map's centroids are no longer kept in
											memory. Compute the map again.
										</p>
									)}
									{similar.error && (
										<p className="text-destructive">{similar.error}</p>
									)}
									<ul>
										{shade?.nearest.map((index) => {
											const point = points[index];
											return (
												<li key={point.id}>
													<button
														type="button"
														className="flex w-full justify-between gap-2 py-0.5 text-left hover:underline"
														onClick={() => setSelected(point)}
													>
														<span className="truncate">{nameOf(point)}</span>
														<span className="shrink-0 font-mono text-xs tabular-nums">
															{shade.similarity[index].toFixed(3)}
														</span>
													</button>
												</li>
											);
										})}
									</ul>
								</div>
							)}
							<div className="space-y-1">
								<h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
									Near-duplicate families ({result.families.length})
								</h3>
								{result.families.length === 0 && (
									<p className="text-muted-foreground">None.</p>
								)}
								{result.families.map((f, index) => (
									<button
										key={f.keep}
										type="button"
										aria-pressed={family === index}
										className="flex w-full justify-between gap-2 py-0.5 text-left hover:underline aria-pressed:font-medium"
										onClick={() => {
											setFamily(family === index ? null : index);
											const keep = byId.get(f.keep);
											if (keep) setSelected(keep);
										}}
									>
										<span className="truncate">
											{f.members.length} documents, keep {f.label[f.keep]}
										</span>
										<span className="shrink-0 font-mono text-xs tabular-nums">
											{Object.values(f.similarity)
												.reduce((a, b) => Math.min(a, b), 1)
												.toFixed(3)}
										</span>
									</button>
								))}
							</div>
						</aside>
					</div>
				</>
			)}
		</section>
	);
}

const SLOTS_SHOWN = SLOT_TOKENS.length;
