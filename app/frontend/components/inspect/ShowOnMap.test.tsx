/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ShowOnMap from "./ShowOnMap";

const map = (ids: string[], stale: boolean | null = false) => ({
	kind: "projection",
	status: "done",
	progress: [],
	queued_at: "t",
	started_at: "t",
	finished_at: "t",
	error: null,
	stale,
	result: {
		points: ids.map((id) => ({ id, x: 0, y: 0, chunks: 1, family: null })),
		families: [],
		left_off: 1,
		generation: "g1",
		neighbours: true,
	},
});

function serve(projection: unknown) {
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) =>
			url.endsWith("/info")
				? Response.json({
						info: {
							embeddings: { vector_dim: 4 },
							tables: [{ name: "chunks", num_rows: 10 }],
						},
					})
				: Response.json({ projection }),
		),
	);
}

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const show = () =>
	render(<ShowOnMap source="alpha" documentId="d1" href="/map?doc=d1" />);

describe("ShowOnMap", () => {
	it("links to the document on a finished map holding it", async () => {
		serve(map(["d0", "d1"]));
		show();
		expect(
			(await screen.findByRole("link", { name: "Show on map" })).getAttribute(
				"href",
			),
		).toBe("/map?doc=d1");
	});

	it("says what is known of why a document is not on the map, and offers to compute it", async () => {
		const cases: [boolean | null, RegExp][] = [
			[false, /it has no usable embedding/],
			[true, /the database changed since the map was computed/],
			[null, /^This document is not on the map\.$/],
		];
		for (const [stale, reason] of cases) {
			serve(map(["d0"], stale));
			show();
			await screen.findByText(reason);
			expect(
				screen.getByRole("button", { name: "Compute again" }),
			).toBeTruthy();
			cleanup();
		}
	});

	it("says when the map cannot be read", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () =>
				Response.json({ error: "backend down" }, { status: 503 }),
			),
		);
		show();
		await screen.findByText("backend down");
	});

	it("offers to compute a map the database does not have", async () => {
		serve(null);
		show();
		await screen.findByText("This database has no map yet.");
		expect(
			screen.getByRole("button", { name: "Compute the map" }),
		).toBeTruthy();
	});
});
