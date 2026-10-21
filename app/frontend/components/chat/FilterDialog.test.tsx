/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import FilterDialog from "./FilterDialog";

const doc = (source: string, n: number) => ({
	id: `${source}-${n}`,
	title: `${source} report ${n}`,
	uri: null,
	source,
});

const alpha = Array.from({ length: 120 }, (_, n) => doc("alpha", n));
let urls: URL[];
let held: Promise<void> | null;
let failAt: number | null;
let gammaDown: boolean;
let claimed: number;

beforeEach(() => {
	urls = [];
	held = null;
	failAt = null;
	gammaDown = false;
	claimed = 120;
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) => {
			const parsed = new URL(url, "http://app");
			urls.push(parsed);
			const limit = Number(parsed.searchParams.get("limit"));
			const offset = Number(parsed.searchParams.get("offset") ?? 0);
			if (limit > 500)
				return Response.json({ error: "limit" }, { status: 400 });
			const q = parsed.searchParams.get("q") ?? "";
			if (q && held) await held;
			if (offset === failAt) {
				failAt = null;
				return Response.json({ error: "read failed" }, { status: 503 });
			}
			const source = parsed.searchParams.get("source");
			const matching = alpha.filter((d) => d.title.includes(q));
			const alphaTotal = q ? matching.length : claimed;
			if (source === "alpha")
				return Response.json({
					documents: matching.slice(offset, offset + limit),
					totals: { alpha: alphaTotal },
					errors: {},
				});
			return Response.json({
				documents: [...matching.slice(0, limit), doc("beta", 0)],
				totals: { alpha: alphaTotal, beta: 1 },
				errors: gammaDown ? { gamma: "database unavailable" } : {},
			});
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const open = () =>
	render(
		<FilterDialog
			open
			onOpenChange={() => {}}
			selection={[]}
			onApply={() => {}}
		/>,
	);

describe("FilterDialog", () => {
	it("lists a page per database, says how many there are, and shows more", async () => {
		const user = userEvent.setup();
		open();
		await screen.findByText("50 of 120");
		expect(screen.getByText("1 of 1")).toBeTruthy();
		await user.click(screen.getByRole("button", { name: "Show more" }));
		// Past the typing delay, which must not undo the page shown since.
		await new Promise((resolve) => setTimeout(resolve, 300));
		await screen.findByText("100 of 120");
		expect(screen.getByText("alpha report 0")).toBeTruthy();
		expect(screen.getByText("alpha report 99")).toBeTruthy();
		expect(screen.getByText("1 of 1")).toBeTruthy();
	});

	it("asks for each further page at the page size, by offset", async () => {
		const user = userEvent.setup();
		open();
		await screen.findByText("50 of 120");
		await user.click(screen.getByRole("button", { name: "Show more" }));
		await screen.findByText("100 of 120");
		await user.click(screen.getByRole("button", { name: "Show more" }));
		await screen.findByText("120 of 120");
		expect(screen.queryByRole("button", { name: "Show more" })).toBeNull();
		const further = urls.filter((url) => url.searchParams.get("source"));
		expect(further.map((url) => url.searchParams.get("offset"))).toEqual([
			"50",
			"100",
		]);
		expect(urls.every((url) => url.searchParams.get("limit") === "50")).toBe(
			true,
		);
	});

	it("asks again for a page that failed", async () => {
		const user = userEvent.setup();
		failAt = 50;
		open();
		await screen.findByText("50 of 120");
		await user.click(screen.getByRole("button", { name: "Show more" }));
		await screen.findByText(/read failed/);
		await user.click(screen.getByRole("button", { name: "Show more" }));
		await screen.findByText("100 of 120");
		expect(urls.map((url) => url.searchParams.get("offset")).slice(-2)).toEqual(
			["50", "50"],
		);
	});

	it("pages each database on its own, whatever another's state", async () => {
		const user = userEvent.setup();
		gammaDown = true;
		open();
		await screen.findByText("50 of 120");
		expect(screen.getByText(/database unavailable/)).toBeTruthy();
		await user.click(screen.getByRole("button", { name: "Show more" }));
		await screen.findByText("100 of 120");
		expect(urls.at(-1)?.searchParams.get("source")).toBe("alpha");
		expect(urls.at(-1)?.searchParams.get("offset")).toBe("50");
	});

	it("stops offering more once a page comes back short", async () => {
		const user = userEvent.setup();
		// The total counts documents the pages no longer reach.
		claimed = 200;
		open();
		await screen.findByText("50 of 200");
		await user.click(screen.getByRole("button", { name: "Show more" }));
		await screen.findByText("100 of 200");
		await user.click(screen.getByRole("button", { name: "Show more" }));
		await screen.findByText("120 of 200");
		expect(screen.queryByRole("button", { name: "Show more" })).toBeNull();
	});

	it("finds documents on the server, showing none of the last search meanwhile", async () => {
		const user = userEvent.setup();
		let release: () => void = () => {};
		held = new Promise((resolve) => {
			release = resolve;
		});
		open();
		await screen.findByText("50 of 120");
		await user.type(
			screen.getByPlaceholderText("Find a document"),
			"report 11",
		);
		await waitFor(() =>
			expect(urls.at(-1)?.searchParams.get("q")).toBe("report 11"),
		);
		expect(screen.queryByText("alpha report 0")).toBeNull();
		expect(screen.getByText("Loading documents…")).toBeTruthy();
		release();
		await screen.findByText("11 of 11");
		expect(screen.queryByRole("button", { name: "Show more" })).toBeNull();
	});
});
