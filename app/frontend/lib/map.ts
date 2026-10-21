export interface Point {
	id: string;
	x: number;
	y: number;
	title: string | null;
	uri: string | null;
	source_id: string | null;
	content_type: string | null;
	chunks: number;
	family: number | null;
}

export type ColourBy = "family" | "source_id" | "content_type";

// Three categorical slots, the rest folded into "other": more hues than three
// stop being told apart on a scatter.
export const SLOTS = 3;

export function categories(
	points: Point[],
	by: ColourBy,
): { slotOf: (point: Point) => number | null; legend: string[] } {
	if (by === "family") {
		return {
			slotOf: (point) => (point.family === null ? null : 0),
			legend: ["near-duplicate family"],
		};
	}
	const frequency = new Map<string, number>();
	for (const point of points) {
		const value = point[by];
		if (value) frequency.set(value, (frequency.get(value) ?? 0) + 1);
	}
	const top = [...frequency.entries()]
		.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
		.slice(0, SLOTS)
		.map(([value]) => value);
	const folded = frequency.size > SLOTS;
	return {
		slotOf: (point) => {
			const value = point[by];
			if (!value) return null;
			const slot = top.indexOf(value);
			return slot === -1 ? SLOTS : slot;
		},
		legend: folded ? [...top, "other"] : top,
	};
}

export interface View {
	k: number;
	tx: number;
	ty: number;
}

// The transform that fits every point in a width × height canvas.
export function fitView(
	points: Point[],
	width: number,
	height: number,
	pad = 28,
): View {
	if (!points.length) return { k: 1, tx: width / 2, ty: height / 2 };
	let [x0, x1, y0, y1] = [
		Number.POSITIVE_INFINITY,
		Number.NEGATIVE_INFINITY,
		Number.POSITIVE_INFINITY,
		Number.NEGATIVE_INFINITY,
	];
	for (const p of points) {
		x0 = Math.min(x0, p.x);
		x1 = Math.max(x1, p.x);
		y0 = Math.min(y0, p.y);
		y1 = Math.max(y1, p.y);
	}
	const spanX = x1 - x0 || 1;
	const spanY = y1 - y0 || 1;
	const k = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY);
	return {
		k,
		tx: width / 2 - ((x0 + x1) / 2) * k,
		ty: height / 2 - ((y0 + y1) / 2) * k,
	};
}

export function toScreen(point: Point, view: View): [number, number] {
	return [point.x * view.k + view.tx, point.y * view.k + view.ty];
}

// Zooms by `factor` keeping the screen point (sx, sy) where it is.
export function zoomAt(
	view: View,
	sx: number,
	sy: number,
	factor: number,
): View {
	return {
		k: view.k * factor,
		tx: sx - (sx - view.tx) * factor,
		ty: sy - (sy - view.ty) * factor,
	};
}

// The point nearest (sx, sy) on screen within `radius` pixels, or null.
export function hit(
	points: Point[],
	view: View,
	sx: number,
	sy: number,
	radius = 10,
): Point | null {
	let best: Point | null = null;
	let bestDistance = radius * radius;
	for (const point of points) {
		const [px, py] = toScreen(point, view);
		const distance = (px - sx) ** 2 + (py - sy) ** 2;
		if (distance <= bestDistance) {
			best = point;
			bestDistance = distance;
		}
	}
	return best;
}

// Each document's rank among the others by similarity to `selected`, from 0
// for the least similar to 1 for the most. Ties share the lower rank.
export function closeness(similarity: number[], selected: number): number[] {
	const others = similarity
		.filter((_, index) => index !== selected)
		.sort((a, b) => a - b);
	const span = Math.max(others.length - 1, 1);
	return similarity.map((value, index) => {
		if (index === selected || others.length < 2) return 1;
		let [low, high] = [0, others.length];
		while (low < high) {
			const middle = (low + high) >> 1;
			if (others[middle] < value) low = middle + 1;
			else high = middle;
		}
		return low / span;
	});
}

// Indices of the `k` documents most similar to `selected`, most similar first.
export function nearestOf(
	similarity: number[],
	selected: number,
	k: number,
): number[] {
	return similarity
		.map((_, index) => index)
		.filter((index) => index !== selected)
		.sort((a, b) => similarity[b] - similarity[a])
		.slice(0, k);
}

export interface Shade {
	similarity: number[];
	close: number[];
	nearest: number[];
	// Point indices, nearest last, so they are drawn on top.
	order: number[];
}

// How every point is shaded while one is selected: only from scores computed
// for the map on screen, and only while the selected document is on it.
export function shading(
	scores: { generation: string; similarity: number[] } | null,
	generation: string | undefined,
	points: Point[],
	selectedId: string | null,
): Shade | null {
	if (!scores || scores.generation !== generation) return null;
	const index = points.findIndex((p) => p.id === selectedId);
	if (index === -1) return null;
	const close = closeness(scores.similarity, index);
	return {
		similarity: scores.similarity,
		close,
		nearest: nearestOf(scores.similarity, index, 8),
		order: points.map((_, i) => i).sort((a, b) => close[a] - close[b]),
	};
}
