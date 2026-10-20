/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DoctorPage from "./DoctorPage";

vi.mock("next/navigation", () => ({
	useParams: () => ({ db: ["alpha"] }),
	useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

const projection = {
	kind: "projection",
	status: "done",
	progress: [],
	queued_at: "t",
	started_at: "t",
	finished_at: "t",
	error: null,
	stale: false,
	result: {
		points: [],
		families: [
			{
				members: ["a", "b"],
				keep: "a",
				similarity: { b: 0.99 },
				label: { a: "doc://a", b: "doc://b" },
			},
		],
		left_off: 0,
		generation: "g1",
		neighbours: true,
	},
};

beforeEach(() => {
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) => {
			const path = new URL(url, "http://app").pathname;
			if (path.endsWith("/databases"))
				return Response.json({
					vision: false,
					databases: [{ name: "alpha", available: true, error: null }],
				});
			if (path.endsWith("/info"))
				return Response.json({
					info: {
						embeddings: { vector_dim: 4 },
						tables: [{ name: "chunks", num_rows: 10 }],
					},
				});
			if (path.endsWith("/projection")) return Response.json({ projection });
			return Response.json({ run: null });
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("DoctorPage", () => {
	it("lists the map's families, links each to the map, and exports them", async () => {
		render(<DoctorPage />);
		const family = await screen.findByRole("link", {
			name: /2 documents, keep doc:\/\/a/,
		});
		expect(family.getAttribute("href")).toBe("/inspect/alpha?view=map&doc=a");
		expect(
			screen
				.getByRole("link", { name: "Download as YAML" })
				.getAttribute("href"),
		).toBe("/api/backend/api/doctor/alpha/duplicates.yaml");
		expect(
			screen.getByRole("link", { name: "Open the map" }).getAttribute("href"),
		).toBe("/inspect/alpha?view=map");
		expect(screen.queryByRole("img")).toBeNull();
	});
});
