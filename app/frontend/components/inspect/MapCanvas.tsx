"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
	fitView,
	hit,
	type Point,
	type Shade,
	toScreen,
	type View,
	zoomAt,
} from "@/lib/map";

export interface Family {
	members: string[];
	keep: string;
	similarity: Record<string, number>;
	label: Record<string, string>;
}

// The colour token a point is filled with, and whether it has no category.
export interface Paint {
	token: string;
	muted: boolean;
}

const RADIUS = 4;

function token(name: string): string {
	return getComputedStyle(document.documentElement)
		.getPropertyValue(name)
		.trim();
}

// The map's canvas: drawing, resize and theme changes, pan, zoom, hover and
// click. What is selected, shaded or highlighted comes from the page.
export default function MapCanvas({
	points,
	families,
	family,
	paint,
	matching,
	selected,
	hovered,
	shade,
	fitKey,
	label,
	onHover,
	onSelect,
}: {
	points: Point[];
	families: Family[];
	family: number | null;
	paint: (point: Point) => Paint;
	matching: Set<string> | null;
	selected: Point | null;
	hovered: Point | null;
	shade: Shade | null;
	// The view refits whenever this changes: a new map, not a re-read of one.
	fitKey: unknown;
	label: (point: Point) => string;
	onHover: (point: Point | null) => void;
	onSelect: (point: Point | null) => void;
}) {
	const canvas = useRef<HTMLCanvasElement>(null);
	const [view, setView] = useState<View>({ k: 1, tx: 0, ty: 0 });
	const drag = useRef<{
		x: number;
		y: number;
		view: View;
		moved: boolean;
	} | null>(null);
	const byId = useMemo(() => new Map(points.map((p) => [p.id, p])), [points]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: a new map has a new fit key
	useEffect(() => {
		const element = canvas.current;
		if (!element || !points.length) return;
		setView(fitView(points, element.clientWidth, element.clientHeight));
	}, [fitKey]);

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
		const resolved = new Map<string, string>();
		const colour = (name: string) => {
			if (!resolved.has(name)) resolved.set(name, token(name));
			return resolved.get(name) as string;
		};
		const ring = token("--destructive");
		const accent = token("--primary");
		const surface = token("--background");
		const near = token("--map-near");
		const far = token("--map-far");

		// Lines from each family member to the one doctor keeps.
		context.lineWidth = 1.5;
		context.strokeStyle = ring;
		families.forEach((f, index) => {
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

		for (const index of shade?.order ?? points.keys()) {
			const point = points[index];
			const [x, y] = toScreen(point, view);
			if (
				x < -RADIUS ||
				y < -RADIUS ||
				x > width + RADIUS ||
				y > height + RADIUS
			)
				continue;
			const { token: fill, muted } = paint(point);
			let alpha = shade ? 1 : muted ? 0.6 : 0.9;
			if (family !== null) alpha = point.family === family ? 1 : 0.12;
			if (matching && !matching.has(point.id)) alpha = Math.min(alpha, 0.1);
			context.globalAlpha = alpha;
			context.fillStyle = shade ? far : colour(fill);
			context.beginPath();
			context.arc(x, y, RADIUS, 0, Math.PI * 2);
			context.fill();
			if (shade) {
				// Cubed, so the colour concentrates on the closest few percent.
				context.globalAlpha = alpha * shade.close[index] ** 3;
				context.fillStyle = near;
				context.fill();
			}
			context.globalAlpha = 1;
			context.strokeStyle = surface;
			context.lineWidth = 1;
			context.stroke();
		}
		context.globalAlpha = 1;
		const circle = (
			point: Point,
			size: number,
			colour: string,
			width: number,
		) => {
			const [x, y] = toScreen(point, view);
			context.strokeStyle = colour;
			context.lineWidth = width;
			context.beginPath();
			context.arc(x, y, size, 0, Math.PI * 2);
			context.stroke();
		};
		for (const index of shade?.nearest ?? [])
			circle(points[index], RADIUS + 3, accent, 1.5);
		if (hovered) circle(hovered, RADIUS + 6, token("--foreground"), 2.5);
		// The selected document is drawn last, larger and in its own colour.
		if (selected) {
			const [x, y] = toScreen(selected, view);
			context.fillStyle = token("--map-selected");
			context.strokeStyle = surface;
			context.lineWidth = 2;
			context.beginPath();
			context.arc(x, y, RADIUS + 3, 0, Math.PI * 2);
			context.fill();
			context.stroke();
			circle(selected, RADIUS + 6, token("--map-selected"), 2);
		}
	}, [
		points,
		families,
		view,
		paint,
		family,
		matching,
		selected,
		hovered,
		shade,
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
	}, []);

	const position = (event: React.PointerEvent<HTMLCanvasElement>) => {
		const rect = event.currentTarget.getBoundingClientRect();
		return [event.clientX - rect.left, event.clientY - rect.top] as const;
	};

	return (
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
						if (Math.abs(x - d.x) + Math.abs(y - d.y) > 3) d.moved = true;
						setView({
							...d.view,
							tx: d.view.tx + x - d.x,
							ty: d.view.ty + y - d.y,
						});
						return;
					}
					onHover(hit(points, view, x, y));
				}}
				onPointerUp={(event) => {
					const [x, y] = position(event);
					if (drag.current && !drag.current.moved) {
						onSelect(hit(points, view, x, y));
					}
					drag.current = null;
				}}
				onPointerLeave={() => onHover(null)}
			/>
			{hovered && (
				<div
					className="pointer-events-none absolute max-w-72 truncate rounded bg-foreground px-2 py-1 text-xs text-background"
					style={{
						left: toScreen(hovered, view)[0] + 12,
						top: toScreen(hovered, view)[1] + 12,
					}}
				>
					{label(hovered)}
				</div>
			)}
		</div>
	);
}
