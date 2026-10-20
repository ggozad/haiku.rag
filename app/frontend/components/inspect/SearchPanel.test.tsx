/** @vitest-environment jsdom */
import {
	cleanup,
	fireEvent,
	render,
	screen,
	waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { searchHref, searchOf } from "@/lib/addresses";
import type { DatabaseEntry } from "@/lib/databases";
import SearchPanel from "./SearchPanel";

// Canvas drawing is out of jsdom's reach: the picked file is the attachment.
vi.mock("@/lib/images", async (actual) => ({
	...(await actual<typeof import("@/lib/images")>()),
	readImage: async (file: File) => ({
		data: "aW1n",
		mimeType: "image/png",
		name: file.name,
	}),
}));

// Answers are held per address for the module's life, so each test searches
// its own addresses.
afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const db = (name: string, supports_image_search = false): DatabaseEntry => ({
	name,
	available: true,
	error: null,
	supports_image_search,
});

const found = (source: string, content: string) => ({
	results: [
		{
			source,
			chunk_id: `${source}-1`,
			document_id: `${source}-doc`,
			document_title: "A document",
			document_uri: "doc://a",
			score: 0.5,
			page_numbers: [],
			headings: null,
			content,
		},
	],
	rendering: content,
});

type Body = { sources: string[]; query?: string; image?: string };

function searching() {
	const bodies: Body[] = [];
	const fetch = vi.fn(async (_url: string, init: RequestInit) => {
		const body = JSON.parse(String(init.body));
		bodies.push(body);
		return Response.json(
			found(body.sources[0], `about ${body.query ?? "the image"}`),
		);
	});
	vi.stubGlobal("fetch", fetch);
	return bodies;
}

let address: string;
const opened = vi.fn();

// Holds the search's address as the page's URL does, and opens it afresh on
// each change, as navigation does.
function Routed({
	databases,
	start = "/search",
}: {
	databases: DatabaseEntry[];
	start?: string;
}) {
	const [href, setHref] = useState(start);
	address = href;
	return (
		<SearchPanel
			key={href}
			databases={databases}
			inputs={searchOf(new URL(href, "http://app").searchParams)}
			onSearch={(inputs) => setHref(searchHref(inputs))}
			onOpen={opened}
		/>
	);
}

const searchButton = () =>
	screen.getByRole("button", { name: "Search" }) as HTMLButtonElement;

async function search(text: string) {
	const user = userEvent.setup();
	const box = screen.getByPlaceholderText("Search the documents");
	await user.clear(box);
	await user.type(box, text);
	await user.click(screen.getByRole("button", { name: "Search" }));
}

describe("SearchPanel", () => {
	it("names the one available database rather than every configured one", async () => {
		const bodies = searching();
		render(<Routed databases={[db("alpha")]} />);
		await search("tea");
		await screen.findByText("about tea");
		expect(bodies[0].sources).toEqual(["alpha"]);
		expect(address).toBe("/search?q=tea&db=alpha&type=hybrid&limit=10");
	});

	it("searches a database that became available, and keeps what the user unticked", async () => {
		const bodies = searching();
		const user = userEvent.setup();
		const { rerender } = render(
			<Routed databases={[db("alpha"), db("beta")]} />,
		);
		await user.click(screen.getByRole("checkbox", { name: "beta" }));
		rerender(<Routed databases={[db("alpha"), db("beta"), db("gamma")]} />);
		await search("tea");
		await screen.findByText("about tea");
		expect(bodies[0].sources).toEqual(["alpha", "gamma"]);
	});

	it("keeps a lone unticked database tickable, and searches nothing without one", async () => {
		searching();
		const user = userEvent.setup();
		const { rerender } = render(
			<Routed databases={[db("alpha"), db("beta")]} />,
		);
		await user.click(screen.getByRole("checkbox", { name: "alpha" }));
		rerender(<Routed databases={[db("alpha")]} />);
		await user.type(screen.getByPlaceholderText("Search the documents"), "tea");
		expect(searchButton().disabled).toBe(true);
		await user.click(screen.getByRole("checkbox", { name: "alpha" }));
		expect(searchButton().disabled).toBe(false);
	});

	it("runs the search its address names, with every input", async () => {
		const bodies = searching();
		render(
			<Routed
				databases={[db("alpha"), db("beta")]}
				start="/search?q=tea&db=beta&type=fts&limit=20&filter=x%3D1"
			/>,
		);
		await screen.findByText("about tea");
		expect(bodies).toEqual([
			{
				query: "tea",
				limit: 20,
				search_type: "fts",
				filter: "x=1",
				sources: ["beta"],
			},
		]);
		expect(
			(screen.getByRole("checkbox", { name: "alpha" }) as HTMLButtonElement)
				.ariaChecked,
		).toBe("false");
		expect(
			(screen.getByPlaceholderText("Search the documents") as HTMLInputElement)
				.value,
		).toBe("tea");
	});

	it("opens a result under its own database and document", async () => {
		searching();
		const user = userEvent.setup();
		render(
			<Routed
				databases={[db("alpha"), db("beta")]}
				start="/search?q=tea&db=beta"
			/>,
		);
		await user.click(await screen.findByText("about tea"));
		expect(opened).toHaveBeenLastCalledWith({
			source: "beta",
			documentId: "beta-doc",
			chunkId: "beta-1",
		});
	});

	it("keeps an image search's answer for its address, until a reload", async () => {
		const bodies = searching();
		const user = userEvent.setup();
		const { unmount } = render(<Routed databases={[db("alpha", true)]} />);
		const file = new File(["png"], "cat.png", { type: "image/png" });
		fireEvent.change(document.getElementById("search-image") as HTMLElement, {
			target: { files: [file] },
		});
		await screen.findByText("cat.png");
		await user.click(searchButton());
		await screen.findByText("about the image");
		expect(bodies).toEqual([
			{
				image: "aW1n",
				limit: 10,
				search_type: "vector",
				filter: null,
				sources: ["alpha"],
			},
		]);
		const left = address;
		unmount();

		render(<Routed databases={[db("alpha", true)]} start={left} />);
		await screen.findByText("about the image");
		expect(screen.getByText("cat.png")).toBeTruthy();
		expect(bodies.length).toBe(1);
	});

	it("keeps the databases its address names once discovery finishes", async () => {
		const bodies = searching();
		const { rerender } = render(
			<Routed databases={[]} start="/search?q=coffee&db=beta" />,
		);
		rerender(
			<Routed
				databases={[db("alpha"), db("beta")]}
				start="/search?q=coffee&db=beta"
			/>,
		);
		await screen.findByText("about coffee");
		expect(
			(screen.getByRole("checkbox", { name: "alpha" }) as HTMLButtonElement)
				.ariaChecked,
		).toBe("false");
		await search("milk");
		await screen.findByText("about milk");
		expect(bodies.map((body) => body.sources)).toEqual([["beta"], ["beta"]]);
	});

	it("runs an address naming no database once the databases are known", async () => {
		const bodies = searching();
		const { rerender } = render(
			<Routed databases={[]} start="/search?q=juice" />,
		);
		rerender(
			<Routed databases={[db("alpha"), db("beta")]} start="/search?q=juice" />,
		);
		await screen.findByText("about juice");
		expect(bodies.map((body) => body.sources)).toEqual([["alpha", "beta"]]);
	});

	it("keeps an image search's answer after another search", async () => {
		const bodies = searching();
		const user = userEvent.setup();
		const { unmount } = render(<Routed databases={[db("alpha", true)]} />);
		fireEvent.change(document.getElementById("search-image") as HTMLElement, {
			target: { files: [new File(["png"], "dog.png", { type: "image/png" })] },
		});
		await screen.findByText("dog.png");
		await user.click(searchButton());
		await screen.findByText("about the image");
		const imageAddress = address;
		await user.click(screen.getByRole("button", { name: "Remove the image" }));
		await search("tea");
		await screen.findByText("about tea");
		unmount();

		render(<Routed databases={[db("alpha", true)]} start={imageAddress} />);
		await screen.findByText("about the image");
		expect(bodies.length).toBe(2);
	});

	it("says it is searching, and when nothing matched", async () => {
		let answer: (response: Response) => void = () => {};
		vi.stubGlobal(
			"fetch",
			vi.fn(
				() =>
					new Promise<Response>((resolve) => {
						answer = resolve;
					}),
			),
		);
		render(<Routed databases={[db("alpha")]} start="/search?q=nothing" />);
		await screen.findByText("Loading…");
		answer(Response.json({ results: [], rendering: "" }));
		await screen.findByText("Nothing matched.");
		expect(screen.queryByText("Loading…")).toBeNull();
	});

	it("cannot search with no database available", async () => {
		searching();
		const user = userEvent.setup();
		render(<Routed databases={[]} />);
		await user.type(screen.getByPlaceholderText("Search the documents"), "tea");
		expect(searchButton().disabled).toBe(true);
	});

	it("aborts a search in flight when the panel goes away", async () => {
		let signal: AbortSignal | undefined;
		vi.stubGlobal(
			"fetch",
			vi.fn((_url: string, init: RequestInit) => {
				signal = init.signal ?? undefined;
				return new Promise<Response>(() => {});
			}),
		);
		const { unmount } = render(<Routed databases={[db("alpha")]} />);
		await search("tea");
		await waitFor(() => expect(signal?.aborted).toBe(false));
		unmount();
		expect(signal?.aborted).toBe(true);
	});
});
