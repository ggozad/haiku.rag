/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import SimilarDocuments from "./SimilarDocuments";

const ids = Array.from({ length: 12 }, (_, i) => `d${i}`);

const job = (
	compared: string[],
	{ stale = false as boolean | null, neighbours = true } = {},
) => ({
	kind: "similarity",
	status: "done",
	progress: [],
	queued_at: "t",
	started_at: "t",
	finished_at: "t",
	error: null,
	stale,
	result: {
		documents: compared.map((id) => ({
			id,
			title: id === "d2" ? "Second report" : null,
			uri: `file:///docs/${id}.md`,
		})),
		families: [],
		left_off: 1,
		generation: "g1",
		neighbours,
	},
});

// Each document's score to d1 is its index over 100, d1's own is 1.
function serve(similarity: unknown, scores = { generation: "g1" }) {
	const fetch = vi.fn(async (url: string) => {
		const path = new URL(url, "http://app").pathname;
		if (path.endsWith("/info"))
			return Response.json({
				info: {
					embeddings: { vector_dim: 4 },
					tables: [{ name: "chunks", num_rows: 10 }],
				},
			});
		if (path.endsWith("/similarity/documents"))
			return Response.json({
				...scores,
				similarity: ids.map((id, i) => (id === "d1" ? 1 : i / 100)),
			});
		return Response.json({ similarity });
	});
	vi.stubGlobal("fetch", fetch);
	return fetch;
}

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const show = async () => {
	render(
		<SimilarDocuments
			source="alpha"
			documentId="d1"
			hrefOf={(id) => `/inspect/alpha/${id}`}
		/>,
	);
	await userEvent
		.setup()
		.click(await screen.findByRole("button", { name: "Similar documents" }));
};

describe("SimilarDocuments", () => {
	it("lists the ten nearest other documents, each linking to its view", async () => {
		const fetch = serve(job(ids));
		await show();
		const links = await screen.findAllByRole("link");
		expect(links.map((link) => link.textContent)).toEqual(
			[11, 10, 9, 8, 7, 6, 5, 4, 3, 2].map(
				(i) =>
					`${i === 2 ? "Second report" : `d${i}.md`}${(i / 100).toFixed(3)}`,
			),
		);
		expect(links[0].getAttribute("href")).toBe("/inspect/alpha/d11");
		expect(links[0].getAttribute("title")).toBe("file:///docs/d11.md");
		expect(
			fetch.mock.calls.some(([url]) =>
				String(url).endsWith(
					"/api/doctor/alpha/similarity/documents?id=d1&generation=g1",
				),
			),
		).toBe(true);
	});

	it("says when the listed scores may be out of date, and offers to compute them again", async () => {
		const cases: [boolean | null, string][] = [
			[true, "Out of date: the database changed"],
			[null, "Whether the database changed is unknown: it cannot be opened"],
		];
		for (const [stale, notice] of cases) {
			serve(job(ids, { stale }));
			await show();
			await screen.findAllByRole("link");
			expect(screen.getByText(notice)).toBeTruthy();
			expect(
				screen.getByRole("button", { name: "Compute again" }),
			).toBeTruthy();
			cleanup();
		}
		serve(job(ids));
		await show();
		await screen.findAllByRole("link");
		expect(screen.queryByRole("button", { name: "Compute again" })).toBeNull();
	});

	it("shows no scores computed by another job", async () => {
		serve(job(ids), { generation: "g0" });
		await show();
		await screen.findByRole("list");
		expect(screen.queryAllByRole("link")).toEqual([]);
	});

	it("says why a document has no similar documents, and offers to compute them", async () => {
		const cases: [ReturnType<typeof job> | null, RegExp, string][] = [
			[null, /^Similarity has not been computed/, "Compute similarity"],
			[
				job(["d0"]),
				/^This document has no usable embedding\.$/,
				"Compute again",
			],
			[
				job(["d0"], { stale: true }),
				/the database changed since similarity was computed/,
				"Compute again",
			],
			[
				job(["d0"], { stale: null }),
				/^This document was not compared\.$/,
				"Compute again",
			],
			[
				job(ids, { neighbours: false }),
				/no longer kept in memory/,
				"Compute again",
			],
		];
		for (const [answer, reason, action] of cases) {
			serve(answer);
			await show();
			await screen.findByText(reason);
			expect(screen.getByRole("button", { name: action })).toBeTruthy();
			await screen.findByText(/Reading the vectors holds/);
			cleanup();
		}
	});

	it("says when the job cannot be read", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () =>
				Response.json({ error: "backend down" }, { status: 503 }),
			),
		);
		render(
			<SimilarDocuments source="alpha" documentId="d1" hrefOf={(id) => id} />,
		);
		await screen.findByText("backend down");
	});
});
