"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import {
	type ColourBy,
	categories,
	fitView,
	hit,
	type Point,
	toScreen,
	type View,
	zoomAt,
} from "@/lib/map";
import { useJson } from "@/lib/useJson";
import { useJob } from "./useJob";

interface Family {
	members: string[];
	keep: string;
	similarity: Record<string, number>;
	label: Record<string, string>;
}

interface MapResult {
	points: Point[];
	families: Family[];
	left_off: number;
	explained_variance: number;
	neighbours: boolean;
}

const SLOT_TOKENS = ["--chart-1", "--chart-2", "--chart-3"];
const RADIUS = 4;

function token(name: string): string {
	return getComputedStyle(document.documentElement)
		.getPropertyValue(name)
		.trim();
}

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

	const canvas = useRef<HTMLCanvasElement>(null);
	const [view, setView] = useState<View>({ k: 1, tx: 0, ty: 0 });
	const [hovered, setHovered] = useState<Point | null>(null);
	const [selected, setSelected] = useState<Point | null>(null);
	const [family, setFamily] = useState<number | null>(null);
	const [colourBy, setColourBy] = useState<ColourBy>("family");
	const [term, setTerm] = useState("");
	const drag = useRef<{
		x: number;
		y: number;
		view: View;
		moved: boolean;
	} | null>(null);

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
	const neighbours = useJson<{
		neighbours: { id: string; similarity: number }[];
	}>(
		selected && result?.neighbours
			? `${base}/projection/neighbours?${new URLSearchParams({ id: selected.id, k: "8" })}`
			: null,
	);
	const byId = useMemo(() => new Map(points.map((p) => [p.id, p])), [points]);

	// Refit when a new map arrives, not when the same map is read again.
	const computedAt = result ? job?.finished_at : null;
	// biome-ignore lint/correctness/useExhaustiveDependencies: a new map has a new finish time
	useEffect(() => {
		const element = canvas.current;
		if (!element || !points.length) return;
		setView(fitView(points, element.clientWidth, element.clientHeight));
	}, [computedAt]);

	const draw = useCallback(() => {
		const element = canvas.current;
		if (!element) return;
		const ratio = window.devicePixelRatio || 1;
		const width = element.clientWidth;
		const height = element.clientHeight;
		element.width = Math.round(width * ratio);
		element.height = Math.round(height * ratio);
		const context = element.getContext("2d");
		if (!context) return;
		context.setTransform(ratio, 0, 0, ratio, 0, 0);
		context.clearRect(0, 0, width, height);
		const slots = SLOT_TOKENS.map(token);
		const other = token("--muted-foreground");
		const ring = token("--destructive");
		const accent = token("--primary");
		const surface = token("--background");

		// Lines from each family member to the one doctor keeps.
		context.lineWidth = 1.5;
		context.strokeStyle = ring;
		result?.families.forEach((f, index) => {
			context.globalAlpha = family === null || family === index ? 0.8 : 0.15;
			const keep = byId.get(f.keep);
			if (!keep) return;
			const [kx, ky] = toScreen(keep, view);
			for (const member of f.members) {
				const p = byId.get(member);
				if (!p || member === f.keep) continue;
				const [mx, my] = toScreen(p, view);
				context.beginPath();
				context.moveTo(kx, ky);
				context.lineTo(mx, my);
				context.stroke();
			}
		});

		for (const point of points) {
			const [x, y] = toScreen(point, view);
			if (
				x < -RADIUS ||
				y < -RADIUS ||
				x > width + RADIUS ||
				y > height + RADIUS
			)
				continue;
			const slot = slotOf(point);
			let alpha = slot === null ? 0.6 : 0.9;
			if (family !== null) alpha = point.family === family ? 1 : 0.12;
			if (matching && !matching.has(point.id)) alpha = Math.min(alpha, 0.1);
			context.globalAlpha = alpha;
			context.fillStyle =
				colourBy === "family"
					? point.family === null
						? other
						: ring
					: slot === null
						? other
						: slot < SLOTS_SHOWN
							? slots[slot]
							: other;
			context.beginPath();
			context.arc(x, y, RADIUS, 0, Math.PI * 2);
			context.fill();
			context.globalAlpha = 1;
			context.strokeStyle = surface;
			context.lineWidth = 1;
			context.stroke();
		}
		context.globalAlpha = 1;
		for (const [point, size, colour] of [
			[selected, RADIUS + 4, accent],
			[hovered, RADIUS + 3, token("--foreground")],
		] as [Point | null, number, string][]) {
			if (!point) continue;
			const [x, y] = toScreen(point, view);
			context.strokeStyle = colour;
			context.lineWidth = 2;
			context.beginPath();
			context.arc(x, y, size, 0, Math.PI * 2);
			context.stroke();
		}
		for (const near of neighbours.data?.neighbours ?? []) {
			const point = byId.get(near.id);
			if (!point) continue;
			const [x, y] = toScreen(point, view);
			context.strokeStyle = accent;
			context.lineWidth = 1.5;
			context.beginPath();
			context.arc(x, y, RADIUS + 3, 0, Math.PI * 2);
			context.stroke();
		}
	}, [
		points,
		view,
		slotOf,
		colourBy,
		family,
		matching,
		selected,
		hovered,
		neighbours.data,
		result,
		byId,
	]);

	useEffect(() => {
		draw();
		const element = canvas.current;
		if (!element) return;
		const observer = new ResizeObserver(draw);
		observer.observe(element);
		const theme = window.matchMedia("(prefers-color-scheme: dark)");
		theme.addEventListener("change", draw);
		return () => {
			observer.disconnect();
			theme.removeEventListener("change", draw);
		};
	}, [draw]);

	// The wheel listener is not passive, so the page does not scroll as it zooms.
	const drawn = result !== null;
	// biome-ignore lint/correctness/useExhaustiveDependencies: the canvas mounts with the map
	useEffect(() => {
		const element = canvas.current;
		if (!element) return;
		const wheel = (event: WheelEvent) => {
			event.preventDefault();
			const rect = element.getBoundingClientRect();
			setView((v) =>
				zoomAt(
					v,
					event.clientX - rect.left,
					event.clientY - rect.top,
					Math.exp(-event.deltaY * 0.0015),
				),
			);
		};
		element.addEventListener("wheel", wheel, { passive: false });
		return () => element.removeEventListener("wheel", wheel);
	}, [drawn]);

	const position = (event: React.PointerEvent<HTMLCanvasElement>) => {
		const rect = event.currentTarget.getBoundingClientRect();
		return [event.clientX - rect.left, event.clientY - rect.top] as const;
	};

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
						{points.length} documents · the two axes explain{" "}
						{(result.explained_variance * 100).toFixed(1)}% of the variance
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
							{legend.map((label, slot) => (
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
						<div className="relative min-w-0">
							<canvas
								ref={canvas}
								role="img"
								aria-label={`Map of ${points.length} documents by the centroid of their chunks`}
								className="h-[34rem] w-full cursor-crosshair touch-none rounded-md border bg-card"
								onPointerDown={(event) => {
									const [x, y] = position(event);
									drag.current = { x, y, view, moved: false };
									event.currentTarget.setPointerCapture(event.pointerId);
								}}
								onPointerMove={(event) => {
									const [x, y] = position(event);
									if (drag.current) {
										const d = drag.current;
										if (Math.abs(x - d.x) + Math.abs(y - d.y) > 3)
											d.moved = true;
										setView({
											...d.view,
											tx: d.view.tx + x - d.x,
											ty: d.view.ty + y - d.y,
										});
										return;
									}
									setHovered(hit(points, view, x, y));
								}}
								onPointerUp={(event) => {
									const [x, y] = position(event);
									if (drag.current && !drag.current.moved) {
										setSelected(hit(points, view, x, y));
									}
									drag.current = null;
								}}
								onPointerLeave={() => setHovered(null)}
							/>
							{hovered && (
								<div
									className="pointer-events-none absolute max-w-72 truncate rounded bg-foreground px-2 py-1 text-xs text-background"
									style={{
										left: toScreen(hovered, view)[0] + 12,
										top: toScreen(hovered, view)[1] + 12,
									}}
								>
									{nameOf(hovered)} · {hovered.chunks} chunks
								</div>
							)}
						</div>
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
									{neighbours.error && (
										<p className="text-destructive">{neighbours.error}</p>
									)}
									<ul>
										{neighbours.data?.neighbours.map((near) => {
											const point = byId.get(near.id);
											return (
												<li key={near.id}>
													<button
														type="button"
														className="flex w-full justify-between gap-2 py-0.5 text-left hover:underline"
														onClick={() => point && setSelected(point)}
													>
														<span className="truncate">
															{point ? nameOf(point) : near.id}
														</span>
														<span className="shrink-0 font-mono text-xs tabular-nums">
															{near.similarity.toFixed(3)}
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
