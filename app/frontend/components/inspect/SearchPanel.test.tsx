/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DatabaseEntry } from "@/lib/databases";
import SearchPanel from "./SearchPanel";

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const db = (name: string): DatabaseEntry => ({
	name,
	available: true,
	error: null,
});

const found = (source: string, content: string) => ({
	results: [
		{
			source,
			chunk_id: `${source}-1`,
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

function searching() {
	const bodies: { sources: string[] }[] = [];
	const fetch = vi.fn(async (_url: string, init: RequestInit) => {
		const body = JSON.parse(String(init.body));
		bodies.push(body);
		return Response.json(found(body.sources[0], `about ${body.query}`));
	});
	vi.stubGlobal("fetch", fetch);
	return bodies;
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
		render(<SearchPanel databases={[db("alpha")]} onOpen={() => {}} />);
		await search("tea");
		await screen.findByText("about tea");
		expect(bodies[0].sources).toEqual(["alpha"]);
	});

	it("searches a database that became available, and keeps what the user unticked", async () => {
		const bodies = searching();
		const user = userEvent.setup();
		const { rerender } = render(
			<SearchPanel databases={[db("alpha"), db("beta")]} onOpen={() => {}} />,
		);
		await user.click(screen.getByRole("checkbox", { name: "beta" }));
		rerender(
			<SearchPanel
				databases={[db("alpha"), db("beta"), db("gamma")]}
				onOpen={() => {}}
			/>,
		);
		await search("tea");
		await screen.findByText("about tea");
		expect(bodies[0].sources).toEqual(["alpha", "gamma"]);
	});

	it("keeps a lone unticked database tickable, and searches nothing without one", async () => {
		searching();
		const user = userEvent.setup();
		const { rerender } = render(
			<SearchPanel databases={[db("alpha"), db("beta")]} onOpen={() => {}} />,
		);
		await user.click(screen.getByRole("checkbox", { name: "alpha" }));
		rerender(<SearchPanel databases={[db("alpha")]} onOpen={() => {}} />);
		await user.type(screen.getByPlaceholderText("Search the documents"), "tea");
		expect(searchButton().disabled).toBe(true);
		await user.click(screen.getByRole("checkbox", { name: "alpha" }));
		expect(searchButton().disabled).toBe(false);
	});

	it("drops a response once the databases searched change", async () => {
		let answer: (response: Response) => void = () => {};
		vi.stubGlobal(
			"fetch",
			vi.fn(
				(_url: string, init: RequestInit) =>
					new Promise<Response>((resolve, reject) => {
						answer = resolve;
						init.signal?.addEventListener("abort", () =>
							reject(new DOMException("aborted", "AbortError")),
						);
					}),
			),
		);
		const { rerender } = render(
			<SearchPanel databases={[db("alpha")]} onOpen={() => {}} />,
		);
		await search("tea");
		rerender(
			<SearchPanel databases={[db("alpha"), db("beta")]} onOpen={() => {}} />,
		);
		answer(Response.json(found("alpha", "stale answer")));
		await waitFor(() => expect(searchButton().disabled).toBe(false));
		expect(screen.queryByText("stale answer")).toBeNull();
	});

	it("cannot search with no database available", async () => {
		searching();
		const user = userEvent.setup();
		render(<SearchPanel databases={[]} onOpen={() => {}} />);
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
		const { unmount } = render(
			<SearchPanel databases={[db("alpha")]} onOpen={() => {}} />,
		);
		await search("tea");
		expect(signal?.aborted).toBe(false);
		unmount();
		expect(signal?.aborted).toBe(true);
	});
});
