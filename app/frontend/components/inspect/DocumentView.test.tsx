/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DocumentView, { type DocumentState, OPENING } from "./DocumentView";

// Holds the view's state as the page's URL does.
function Stateful({
	source,
	documentId,
	onOpenChunk,
}: {
	source: string;
	documentId: string;
	onOpenChunk: (chunkId: string) => void;
}) {
	const [state, setState] = useState<DocumentState>(OPENING);
	return (
		<DocumentView
			source={source}
			documentId={documentId}
			state={state}
			onState={(change) => setState((prev) => ({ ...prev, ...change }))}
			onOpenChunk={onOpenChunk}
		/>
	);
}

const item = (n: number, label = "text", chunk = `c${n}`) => ({
	self_ref: `#/texts/${n}`,
	label,
	text: `Item ${n}`,
	page_numbers: [1],
	heading_level: 0,
	chunk_ids: chunk ? [chunk] : [],
});

const chunk = (n: number) => ({
	id: `c${n}`,
	order: n,
	content: `Chunk ${n} text`,
	metadata: { page_numbers: [1] },
});

let requested: string[];
let scrolled: string[];

function serve(documentId: string) {
	return vi.fn(async (url: string) => {
		requested.push(url);
		const { pathname, searchParams } = new URL(url, "http://app");
		const path = pathname.replace(/^.*\/documents\/[^/]+/, "");
		const answers: Record<string, unknown> = {
			"": {
				id: documentId,
				title: `Title of ${documentId}`,
				uri: `doc://${documentId}`,
				metadata: {},
				created_at: "t",
				updated_at: "t",
			},
			"/items": {
				items: [item(0, "section_header"), item(1), item(2, "text", "")],
			},
			"/outline": {
				outline: [
					{
						self_ref: "#/texts/0",
						level: 1,
						title: "Second section",
						page_numbers: [2],
						item_range: [2, 3],
						chunk_ids: ["c1"],
						children: [],
					},
				],
			},
			"/chunks": {
				total: 120,
				chunks: Array.from({ length: 50 }, (_, n) =>
					chunk(n + Number(searchParams.get("offset"))),
				),
			},
			"/pages": { pages: [{ page: 3, chunk_ids: ["c7"] }] },
			"/pictures": {
				pictures: [
					{
						self_ref: "#/pictures/0",
						caption: null,
						description: null,
						page_numbers: [],
						chunk_id: "c9",
					},
				],
			},
		};
		return Response.json(answers[path]);
	});
}

beforeEach(() => {
	requested = [];
	scrolled = [];
	Element.prototype.scrollIntoView = function (this: Element) {
		scrolled.push(this.id);
	};
	vi.stubGlobal("fetch", serve("d1"));
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const reads = (suffix: string) =>
	requested.filter((url) =>
		new URL(url, "http://app").pathname.endsWith(suffix),
	).length;

describe("DocumentView", () => {
	it("shows the content, and opens the chunk an item belongs to", async () => {
		const opened = vi.fn();
		const user = userEvent.setup();
		render(<Stateful source="alpha" documentId="d1" onOpenChunk={opened} />);
		await screen.findByText("Title of d1");
		await user.click(await screen.findByText("Item 1"));
		expect(opened).toHaveBeenCalledWith("c1");
		expect(screen.getByText("Item 2").closest("button")).toBeNull();
	});

	it("jumps from the outline to the heading's item in the content", async () => {
		const user = userEvent.setup();
		render(<Stateful source="alpha" documentId="d1" onOpenChunk={() => {}} />);
		await screen.findByText("Item 0");
		await user.click(screen.getByRole("tab", { name: "Outline" }));
		await user.click(await screen.findByText("Second section"));
		await waitFor(() => expect(scrolled).toEqual(["item-2"]));
		expect(
			screen
				.getByRole("tab", { name: "Content" })
				.getAttribute("aria-selected"),
		).toBe("true");
	});

	it("pages through the chunks, and keeps the page across tabs", async () => {
		const opened = vi.fn();
		const user = userEvent.setup();
		render(<Stateful source="alpha" documentId="d1" onOpenChunk={opened} />);
		await user.click(screen.getByRole("tab", { name: "Chunks" }));
		await screen.findByText("Chunk 0 text");
		await user.click(screen.getByRole("button", { name: "Next" }));
		await screen.findByText("Chunk 50 text");
		expect(screen.getByText(/51–100 of/)).toBeTruthy();
		await user.click(screen.getByRole("tab", { name: "Content" }));
		await user.click(screen.getByRole("tab", { name: "Chunks" }));
		await user.click(await screen.findByText("Chunk 50 text"));
		expect(opened).toHaveBeenCalledWith("c50");
	});

	it("draws a page with the chunks on it", async () => {
		const opened = vi.fn();
		const user = userEvent.setup();
		render(<Stateful source="alpha" documentId="d1" onOpenChunk={opened} />);
		await user.click(screen.getByRole("tab", { name: "Pages" }));
		await user.click(await screen.findByRole("button", { name: "3" }));
		const page = screen.getByAltText("Page 3 with its chunks drawn");
		expect(page.getAttribute("src")).toContain("/documents/d1/pages/3");
		await user.click(screen.getByText("c7"));
		expect(opened).toHaveBeenCalledWith("c7");
	});

	it("lists the pictures, saying when one has no caption", async () => {
		const opened = vi.fn();
		const user = userEvent.setup();
		render(<Stateful source="alpha" documentId="d1" onOpenChunk={opened} />);
		await user.click(screen.getByRole("tab", { name: "Pictures" }));
		await screen.findByText("No caption or description.");
		await user.click(screen.getByRole("button", { name: "Open its chunk" }));
		expect(opened).toHaveBeenCalledWith("c9");
	});

	it("opens the tab, chunk page, drawn page and item its state names", async () => {
		const view = (state: Partial<DocumentState>) =>
			render(
				<DocumentView
					source="alpha"
					documentId="d1"
					state={{ ...OPENING, ...state }}
					onState={() => {}}
					onOpenChunk={() => {}}
				/>,
			);
		let shown = view({ tab: "chunks", offset: 50 });
		await screen.findByText("Chunk 50 text");
		expect(screen.getByText(/51–100 of/)).toBeTruthy();
		shown.unmount();

		shown = view({ tab: "pages", page: 3 });
		await screen.findByAltText("Page 3 with its chunks drawn");
		shown.unmount();

		view({ tab: "content", item: 2 });
		await waitFor(() => expect(scrolled).toEqual(["item-2"]));
		expect(
			document.getElementById("item-2")?.getAttribute("aria-current"),
		).toBe("location");
		expect(
			document.getElementById("item-1")?.hasAttribute("aria-current"),
		).toBe(false);
	});

	it("reads the shared views once, and a tab's own view only when shown", async () => {
		const user = userEvent.setup();
		render(<Stateful source="alpha" documentId="d1" onOpenChunk={() => {}} />);
		await screen.findByText("Item 0");
		expect(reads("/pages") + reads("/pictures")).toBe(0);
		await user.click(screen.getByRole("tab", { name: "Outline" }));
		await user.click(screen.getByRole("tab", { name: "Chunks" }));
		await user.click(screen.getByRole("tab", { name: "Content" }));
		expect(reads("/items")).toBe(1);
		expect(reads("/outline")).toBe(1);
		expect(reads("/chunks")).toBe(1);
	});

	it("reads the database and document it names, and follows a change of either", async () => {
		const user = userEvent.setup();
		const { rerender } = render(
			<Stateful source="alpha" documentId="d1" onOpenChunk={() => {}} />,
		);
		await screen.findByText("Title of d1");
		expect(
			requested.every((url) => url.includes("/inspect/alpha/documents/d1")),
		).toBe(true);
		await user.click(screen.getByRole("tab", { name: "Pictures" }));
		const picture = await screen.findByRole("img");
		const query = new URL(picture.getAttribute("src") ?? "", "http://app")
			.searchParams;
		expect([query.get("source"), query.get("document_id")]).toEqual([
			"alpha",
			"d1",
		]);

		requested = [];
		vi.stubGlobal("fetch", serve("d2"));
		rerender(<Stateful source="beta" documentId="d2" onOpenChunk={() => {}} />);
		await screen.findByText("Title of d2");
		expect(requested.length).toBeGreaterThan(0);
		expect(
			requested.every((url) => url.includes("/inspect/beta/documents/d2")),
		).toBe(true);
	});

	it("says when a document has no pictures, or the listing fails", async () => {
		const user = userEvent.setup();
		const pictures = (respond: () => Response) =>
			vi.fn(async (url: string) =>
				url.endsWith("/pictures") ? respond() : serve("d1")(url),
			);
		vi.stubGlobal(
			"fetch",
			pictures(() => Response.json({ pictures: [] })),
		);
		const { unmount } = render(
			<Stateful source="alpha" documentId="d1" onOpenChunk={() => {}} />,
		);
		await user.click(screen.getByRole("tab", { name: "Pictures" }));
		await screen.findByText("This document has no pictures.");
		unmount();

		vi.stubGlobal(
			"fetch",
			pictures(() =>
				Response.json({ error: "no such document" }, { status: 404 }),
			),
		);
		render(<Stateful source="alpha" documentId="d1" onOpenChunk={() => {}} />);
		await user.click(screen.getByRole("tab", { name: "Pictures" }));
		await screen.findByText("no such document");
	});
});
