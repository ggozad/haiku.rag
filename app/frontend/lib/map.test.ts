import { describe, expect, it } from "vitest";
import {
	categories,
	closeness,
	fitView,
	hit,
	nearestOf,
	type Point,
	toScreen,
	zoomAt,
} from "./map";

const point = (
	id: string,
	x: number,
	y: number,
	extra: Partial<Point> = {},
): Point => ({
	id,
	x,
	y,
	title: null,
	uri: null,
	source_id: null,
	content_type: null,
	chunks: 3,
	family: null,
	...extra,
});

describe("categories", () => {
	it("gives the three most frequent values a slot and folds the rest", () => {
		const points = ["a", "a", "b", "b", "c", "d", "e"].map((s, n) =>
			point(`p${n}`, 0, 0, { source_id: s }),
		);
		points.push(point("none", 0, 0));
		const { slotOf, legend } = categories(points, "source_id");
		expect(legend).toEqual(["a", "b", "c", "other"]);
		expect(points.map(slotOf)).toEqual([0, 0, 1, 1, 2, 3, 3, null]);
	});

	it("needs no other slot for three values, and colours family members", () => {
		const points = [
			point("a", 0, 0, { content_type: "x", family: 2 }),
			point("b", 0, 0, { content_type: "y" }),
		];
		expect(categories(points, "content_type").legend).toEqual(["x", "y"]);
		const family = categories(points, "family");
		expect(points.map(family.slotOf)).toEqual([0, null]);
	});
});

describe("view", () => {
	it("fits points inside the canvas, centred", () => {
		const points = [point("a", -1, -1), point("b", 1, 1)];
		const view = fitView(points, 200, 100, 10);
		const [ax, ay] = toScreen(points[0], view);
		const [bx, by] = toScreen(points[1], view);
		expect((ax + bx) / 2).toBeCloseTo(100);
		expect((ay + by) / 2).toBeCloseTo(50);
		expect(by - ay).toBeCloseTo(80);
		expect(fitView([], 200, 100)).toEqual({ k: 1, tx: 100, ty: 50 });
		expect(fitView([point("a", 3, 3)], 200, 100).k).toBeGreaterThan(0);
		const many = Array.from({ length: 200_000 }, (_, n) =>
			point(`p${n}`, n, -n),
		);
		expect(fitView(many, 200, 100).k).toBeGreaterThan(0);
	});

	it("zooms about the cursor", () => {
		const view = { k: 2, tx: 10, ty: 20 };
		const zoomed = zoomAt(view, 50, 60, 3);
		const p = point("a", (50 - 10) / 2, (60 - 20) / 2);
		expect(toScreen(p, zoomed)).toEqual([50, 60]);
	});

	it("hits the nearest point within the radius", () => {
		const view = { k: 1, tx: 0, ty: 0 };
		const points = [point("a", 10, 10), point("b", 14, 10)];
		expect(hit(points, view, 13, 10)?.id).toBe("b");
		expect(hit(points, view, 100, 100)).toBeNull();
	});
});

describe("closeness", () => {
	it("ranks the others by similarity, ties together, the selected at one", () => {
		const similarity = [0.2, 1, 0.9, 0.5, 0.9, 0.1];
		expect(closeness(similarity, 1)).toEqual([0.25, 1, 0.75, 0.5, 0.75, 0]);
		expect(closeness([1], 0)).toEqual([1]);
		expect(closeness([1, 0.3], 0)).toEqual([1, 1]);
	});

	it("lists the k nearest others, most similar first", () => {
		const similarity = [0.2, 1, 0.9, 0.5, 0.95];
		expect(nearestOf(similarity, 1, 3)).toEqual([4, 2, 3]);
		expect(nearestOf(similarity, 1, 10)).toHaveLength(4);
	});
});
