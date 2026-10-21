"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
	doctorBase,
	FamilyList,
	MapControls,
	useMapJob,
} from "@/components/doctor/MapJob";
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
import MapCanvas, { type Paint } from "./MapCanvas";

const SLOT_TOKENS = ["--chart-1", "--chart-2", "--chart-3"];
const OTHER = "--muted-foreground";

const nameOf = (point: Point) => point.title || point.uri || point.id;

// The map of a database's documents. The selection is its owner's, held by
// id, so a recomputed map resolves it against its own points.
export default function CorpusMap({
	source,
	ready,
	selectedId,
	onSelect,
	documentHref,
}: {
	source: string;
	// The memory figure is shown, so the operator can decide.
	ready: boolean;
	selectedId: string | null;
	onSelect: (id: string | null) => void;
	documentHref: (id: string) => string;
}) {
	const base = doctorBase(source);
	const map = useMapJob(source);
	const { result, computedAt } = map;
	const points = result?.points ?? [];

	const [hoveredId, setHoveredId] = useState<string | null>(null);
	const setHovered = (point: Point | null) => setHoveredId(point?.id ?? null);
	const setSelected = (point: Point | null) => onSelect(point?.id ?? null);
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
	// biome-ignore lint/correctness/useExhaustiveDependencies: a new map has a new finish time
	useEffect(() => {
		setFamily(null);
		if (!result) return;
		if (selectedId && !byId.has(selectedId)) onSelect(null);
		setHoveredId((id) => (id && byId.has(id) ? id : null));
	}, [computedAt]);

	return (
		<section className="space-y-3">
			<MapControls map={map} ready={ready} />
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
										<p className="flex items-baseline gap-1.5 break-words font-medium">
											<span
												aria-hidden
												className="inline-block size-2.5 shrink-0 rounded-full"
												style={{ background: "var(--map-selected)" }}
											/>
											{nameOf(selected)}
										</p>
										<p className="text-xs text-muted-foreground">
											{selected.chunks} chunks
											{selected.family !== null &&
												` · family ${selected.family + 1}`}
										</p>
										<Link
											className="text-xs underline"
											href={documentHref(selected.id)}
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
														onMouseEnter={() => setHovered(point)}
														onMouseLeave={() => setHovered(null)}
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
							<FamilyList
								source={source}
								families={result.families}
								active={family}
								onPick={(index) => {
									setFamily(family === index ? null : index);
									const keep = byId.get(result.families[index].keep);
									if (keep) setSelected(keep);
								}}
							/>
						</aside>
					</div>
				</>
			)}
		</section>
	);
}

const SLOTS_SHOWN = SLOT_TOKENS.length;
