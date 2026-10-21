/** @vitest-environment jsdom */
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SearchOverlay, { openSearch } from "./SearchOverlay";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

beforeEach(() => {
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init: RequestInit = {}) => {
			if (url.endsWith("/databases"))
				return Response.json({
					vision: false,
					databases: [{ name: "alpha", available: true, error: null }],
				});
			const body = JSON.parse(String(init.body));
			return Response.json({
				results: [
					{
						source: "alpha",
						chunk_id: "c1",
						document_id: "d1",
						document_title: "A document",
						document_uri: null,
						score: 0.5,
						page_numbers: [],
						headings: null,
						content: `about ${body.query}`,
					},
				],
				rendering: "",
			});
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	push.mockReset();
});

async function search(text: string) {
	const user = userEvent.setup();
	await user.type(
		await screen.findByPlaceholderText("Search the documents"),
		text,
	);
	await user.keyboard("{Enter}");
	return user;
}

describe("SearchOverlay", () => {
	it("opens on Ctrl+K and on Cmd+K, with the query focused", async () => {
		const user = userEvent.setup();
		render(<SearchOverlay />);
		expect(screen.queryByRole("dialog")).toBeNull();
		await user.keyboard("{Control>}k{/Control}");
		await screen.findByRole("dialog");
		await waitFor(() =>
			expect(document.activeElement).toBe(
				screen.getByPlaceholderText("Search the documents"),
			),
		);
		await user.keyboard("{Escape}");
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
		await user.keyboard("{Meta>}k{/Meta}");
		await screen.findByRole("dialog");
	});

	it("shows results in place, and opens a result's chunk page", async () => {
		render(<SearchOverlay />);
		act(() => openSearch());
		const user = await search("tea");
		expect(
			screen.getByRole("link", { name: "Open as a page" }).getAttribute("href"),
		).toBe("/search?q=tea&db=alpha&type=hybrid&limit=10");
		await user.click(await screen.findByText("about tea"));
		expect(push).toHaveBeenCalledWith("/inspect/alpha/d1/chunks/c1");
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
	});

	it("keeps the last search when opened again", async () => {
		render(<SearchOverlay />);
		act(() => openSearch());
		const user = await search("milk");
		await screen.findByText("about milk");
		await user.keyboard("{Escape}");
		act(() => openSearch());
		await screen.findByText("about milk");
	});
});
