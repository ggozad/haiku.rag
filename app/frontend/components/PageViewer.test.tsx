/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PdfLoading } from "@/lib/pdf";
import PageViewer, { GroundingPages } from "./PageViewer";

const openPdf = vi.hoisted(() => vi.fn());
vi.mock("@/lib/pdf", () => ({ openPdf }));

const rendered: number[] = [];
const destroyed = vi.fn();

function fakePdf() {
	return {
		getPage: async (page: number) => ({
			getViewport: ({ scale }: { scale: number }) => ({
				width: 612 * scale,
				height: 792 * scale,
			}),
			render: () => {
				rendered.push(page);
				return { promise: Promise.resolve(), cancel: () => {} };
			},
		}),
		loadingTask: { destroy: destroyed },
	};
}

const pages = [
	{
		page: 1,
		width: 612,
		height: 792,
		boxes: [{ l: 61.2, t: 712.8, r: 306, b: 633.6, matched: true }],
	},
	{
		page: 2,
		width: 792,
		height: 612,
		boxes: [{ l: 79.2, t: 550.8, r: 396, b: 367.2, matched: false }],
	},
];

let scrolled: string[];

beforeEach(() => {
	rendered.length = 0;
	scrolled = [];
	openPdf.mockReset().mockImplementation(async () => fakePdf());
	destroyed.mockReset();
	Element.prototype.scrollIntoView = function (this: Element) {
		scrolled.push(this.getAttribute("aria-label") ?? "");
	};
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("PageViewer", () => {
	it("renders a PDF's pages from the original and draws the boxes on them", async () => {
		render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: true }}
				pages={pages}
				focus={2}
				onStale={() => {}}
			/>,
		);

		await waitFor(() => expect(rendered.sort()).toEqual([1, 2]));
		expect(openPdf).toHaveBeenCalledWith(
			"/api/backend/api/inspect/alpha/documents/d1/original?sha256=m",
			expect.objectContaining({ signal: expect.any(AbortSignal) }),
		);

		const sideways = screen.getByRole("img", { name: "Page 2" });
		expect(sideways.style.aspectRatio).toBe("792 / 612");
		const box = sideways.querySelector<HTMLElement>("[data-matched]");
		expect(box?.dataset.matched).toBe("false");
		expect([box?.style.left, box?.style.top, box?.style.width]).toEqual([
			"10%",
			"10%",
			"40%",
		]);
		expect(box?.style.height).toBe("30%");
		expect(scrolled).toEqual(["Page 2"]);
	});

	it("reports a replaced original instead of opening it", async () => {
		openPdf.mockImplementation(async (_url, loading: PdfLoading) => {
			loading.onStale();
			return null;
		});
		const onStale = vi.fn();
		render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: true }}
				pages={pages}
				onStale={onStale}
			/>,
		);
		await waitFor(() => expect(onStale).toHaveBeenCalledOnce());
		expect(rendered).toEqual([]);
	});

	it("says when the original cannot be read", async () => {
		openPdf.mockRejectedValue(new Error("The original answered 404."));
		render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: true }}
				pages={pages}
				onStale={() => {}}
			/>,
		);
		await screen.findByText("The original answered 404.");
	});

	it("says when a later read of the original fails", async () => {
		openPdf.mockImplementation(async (_url, loading: PdfLoading) => {
			setTimeout(() => loading.onError("The original answered 500."));
			return fakePdf();
		});
		render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: true }}
				pages={pages}
				onStale={() => {}}
			/>,
		);
		await screen.findByText("The original answered 500.");
	});

	it("ignores what the original reports after it unmounts", async () => {
		let loading: PdfLoading | undefined;
		openPdf.mockImplementation(async (_url, given: PdfLoading) => {
			loading = given;
			return fakePdf();
		});
		const onStale = vi.fn();
		const shown = render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: true }}
				pages={pages}
				onStale={onStale}
			/>,
		);
		await waitFor(() => expect(loading).toBeDefined());
		shown.unmount();
		loading?.onStale();
		loading?.onError("late");
		expect(onStale).not.toHaveBeenCalled();
	});

	it("closes the PDF it opened when it unmounts", async () => {
		const shown = render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: true }}
				pages={pages}
				onStale={() => {}}
			/>,
		);
		await waitFor(() => expect(rendered.length).toBe(2));
		shown.unmount();
		expect(destroyed).toHaveBeenCalledOnce();
	});

	it("says when a page fails to render, and not when its render is cancelled", async () => {
		const cancelled = Object.assign(new Error("cancelled"), {
			name: "RenderingCancelledException",
		});
		openPdf.mockImplementation(async () => ({
			...fakePdf(),
			getPage: async (page: number) => {
				if (page === 2) throw new Error("bad page");
				return {
					getViewport: () => ({ width: 612, height: 792 }),
					render: () => ({
						promise: Promise.reject(cancelled),
						cancel: () => {},
					}),
				};
			},
		}));
		render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: true }}
				pages={pages}
				onStale={() => {}}
			/>,
		);
		const failed = screen.getByRole("img", { name: "Page 2" });
		await waitFor(() =>
			expect(failed.textContent).toContain(
				"Page 2 could not be rendered: bad page",
			),
		);
		expect(screen.getByRole("img", { name: "Page 1" }).textContent).toBe("");
	});

	it("shows an image's pages as the backend renders them", async () => {
		const fetched = vi.fn();
		vi.stubGlobal("fetch", fetched);
		const onStale = vi.fn();
		render(
			<PageViewer
				source="alpha"
				documentId="d1"
				original={{ sha256: "m", pdf: false }}
				pages={pages.slice(0, 1)}
				onStale={onStale}
			/>,
		);
		const image = screen
			.getByRole("img", { name: "Page 1" })
			.querySelector("img") as HTMLImageElement;
		expect(image.getAttribute("src")).toBe(
			"/api/backend/api/inspect/alpha/documents/d1/pages/1?sha256=m",
		);
		image.dispatchEvent(new Event("error"));
		expect(onStale).toHaveBeenCalledOnce();
		expect(fetched).not.toHaveBeenCalled();
		expect(openPdf).not.toHaveBeenCalled();
	});
});

describe("GroundingPages", () => {
	const grounded = (sha256: string) => ({
		source: "beta",
		document_id: "d1",
		original: { sha256, pdf: true },
		pages: pages.slice(0, 1),
		message: null,
	});

	it("draws the chunks on the original the answer names, read again when it changed", async () => {
		const answers = [grounded("old"), grounded("new")];
		const fetched = vi.fn(async (_url: string) =>
			Response.json(answers.shift()),
		);
		vi.stubGlobal("fetch", fetched);
		openPdf.mockImplementation(async (url: string, loading: PdfLoading) => {
			if (!url.endsWith("sha256=old")) return fakePdf();
			loading.onStale();
			return null;
		});
		render(<GroundingPages url="/api/backend/api/visualize/c1?source=beta" />);

		await waitFor(() => expect(rendered).toEqual([1]));
		expect(fetched.mock.calls.map(([url]) => url)).toEqual([
			"/api/backend/api/visualize/c1?source=beta",
			"/api/backend/api/visualize/c1?source=beta",
		]);
		expect(openPdf.mock.calls.map(([url]) => url)).toEqual([
			"/api/backend/api/inspect/beta/documents/d1/original?sha256=old",
			"/api/backend/api/inspect/beta/documents/d1/original?sha256=new",
		]);
		expect(
			screen
				.getByRole("img", { name: "Page 1" })
				.querySelector("[data-matched=true]"),
		).not.toBeNull();
	});

	it("says when the boxes cannot be read again after the original changed", async () => {
		let reads = 0;
		vi.stubGlobal(
			"fetch",
			vi.fn(async (url: string) => {
				if (url.startsWith("/api/backend/api/visualize/"))
					return reads++
						? Response.json({ detail: "chunk not found" }, { status: 404 })
						: Response.json(grounded("old"));
				throw new Error(`unexpected ${url}`);
			}),
		);
		openPdf.mockImplementation(async (_url, loading: PdfLoading) => {
			loading.onStale();
			return null;
		});
		render(<GroundingPages url="/api/backend/api/visualize/c1?source=beta" />);
		await screen.findByText(/chunk not found/);
		expect(screen.queryByRole("img", { name: "Page 1" })).toBeNull();
	});

	it("says when nothing can be drawn", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () =>
				Response.json({
					source: "beta",
					document_id: null,
					original: null,
					pages: [],
					message: "No visual grounding available",
				}),
			),
		);
		render(<GroundingPages url="/api/backend/api/visualize/c1" />);
		await screen.findByText("No visual grounding available");
	});

	it("says why the answer failed", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () =>
				Response.json({ detail: "chunk not found" }, { status: 404 }),
			),
		);
		render(<GroundingPages url="/api/backend/api/visualize/c1" />);
		await screen.findByText(/chunk not found/);
	});
});
