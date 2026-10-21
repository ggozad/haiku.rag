/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ChunkView from "./ChunkView";

const row = (order: number) => ({
	id: `c${order}`,
	document_id: "d1",
	order,
	content: `Chunk ${order} text`,
	metadata: { page_numbers: [2] },
});

let requested: string[];

beforeEach(() => {
	requested = [];
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) => {
			requested.push(url);
			const { pathname, searchParams } = new URL(url, "http://app");
			if (pathname.endsWith("/similar"))
				return Response.json({
					results: [
						{
							source: "beta",
							chunk_id: "x9",
							document_id: "dx",
							document_title: "Elsewhere",
							document_uri: null,
							score: 0.9,
							content: "A similar passage",
						},
					],
				});
			if (pathname.includes("/chunks/c"))
				return Response.json(row(Number(pathname.split("/c").pop())));
			if (pathname.endsWith("/items"))
				return Response.json({
					items: [
						{ self_ref: "#/texts/0", chunk_ids: ["c0"] },
						{ self_ref: "#/texts/1", chunk_ids: ["c0", "c1"] },
						{ self_ref: "#/texts/2", chunk_ids: ["c1"] },
					],
				});
			if (pathname.endsWith("/documents/d1/chunks")) {
				const offset = Number(searchParams.get("offset"));
				const total = 3;
				return Response.json({
					total,
					chunks: [0, 1, 2]
						.slice(offset, offset + Number(searchParams.get("limit")))
						.map(row),
				});
			}
			return Response.json({ error: "unexpected" }, { status: 404 });
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const show = (chunkId: string, view = "details", onOpen = vi.fn()) =>
	render(
		<ChunkView
			chunk={{ source: "alpha", documentId: "d1", chunkId }}
			view={view}
			onView={() => {}}
			onOpen={onOpen}
			chunkHref={(id) => `/chunk/${id}`}
			documentHref={(item) => `/document?item=${item}`}
		/>,
	);

describe("ChunkView", () => {
	it("shows the chunk with links to its neighbours in the document", async () => {
		show("c1");
		await screen.findByText("Chunk 1 text");
		expect(
			(
				await screen.findByRole("link", { name: "Previous chunk" })
			).getAttribute("href"),
		).toBe("/chunk/c0");
		expect(
			screen.getByRole("link", { name: "Next chunk" }).getAttribute("href"),
		).toBe("/chunk/c2");
		expect(screen.getByText("2 of 3")).toBeTruthy();
		expect(screen.getByRole("heading", { name: "Chunk 2" })).toBeTruthy();
	});

	it("has no neighbour past either end", async () => {
		show("c0");
		await screen.findByRole("link", { name: "Next chunk" });
		expect(screen.queryByRole("link", { name: "Previous chunk" })).toBeNull();
		cleanup();
		show("c2");
		await screen.findByRole("link", { name: "Previous chunk" });
		expect(screen.queryByRole("link", { name: "Next chunk" })).toBeNull();
	});

	it("links to the first item the chunk holds in the document", async () => {
		show("c1");
		expect(
			(
				await screen.findByRole("link", { name: "Show in the document" })
			).getAttribute("href"),
		).toBe("/document?item=1");
	});

	it("opens the view its address names, and a similar chunk under its own database", async () => {
		const opened = vi.fn();
		const user = userEvent.setup();
		show("c1", "similar", opened);
		await user.click(await screen.findByText("A similar passage"));
		expect(opened).toHaveBeenCalledWith({
			source: "beta",
			documentId: "dx",
			chunkId: "x9",
		});
		expect(
			screen
				.getByRole("tab", { name: "Similar" })
				.getAttribute("aria-selected"),
		).toBe("true");
	});
});
