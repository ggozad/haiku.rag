/** @vitest-environment jsdom */
import {
	act,
	cleanup,
	fireEvent,
	render,
	screen,
	waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fitView, toScreen } from "@/lib/map";
import CorpusMap from "./CorpusMap";

const point = (id: string, family: number | null) => ({
	id,
	x: Number(id.charCodeAt(0)),
	y: 0,
	title: `Document ${id}`,
	uri: `doc://${id}`,
	source_id: null,
	content_type: null,
	chunks: 3,
	family,
});

const finished = (generation: string) => ({
	kind: "projection",
	status: "done",
	progress: [],
	queued_at: generation,
	started_at: generation,
	finished_at: generation,
	error: null,
	stale: false,
	result: {
		points: [point("a", 0), point("b", 0), point("c", null)],
		families: [
			{
				members: ["a", "b"],
				keep: "a",
				similarity: { b: 0.99 },
				label: { a: "doc://a", b: "doc://b" },
			},
		],
		left_off: 0,
		generation,
		neighbours: true,
		versions: {},
	},
});

interface Asked {
	generation: string;
	answer: (similarity: number[]) => void;
}

let current: ReturnType<typeof finished>;
let asked: Asked[];

beforeEach(() => {
	asked = [];
	current = finished("g1");
	vi.stubGlobal(
		"matchMedia",
		() => ({ addEventListener() {}, removeEventListener() {} }) as never,
	);
	HTMLCanvasElement.prototype.getContext = () => null;
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init?: RequestInit) => {
			const { pathname, searchParams } = new URL(url, "http://app");
			if (pathname.endsWith("/projection/similarity")) {
				const generation = searchParams.get("generation") ?? "";
				return new Promise<Response>((resolve) =>
					asked.push({
						generation,
						answer: (similarity) =>
							resolve(Response.json({ generation, similarity })),
					}),
				);
			}
			if (init?.method === "POST") {
				current = finished("g2");
				return Response.json({
					...finished("g2"),
					status: "queued",
					result: null,
				});
			}
			return Response.json({ projection: current });
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("CorpusMap", () => {
	it("keeps a selection through a recompute and shows only the new map's scores", async () => {
		const user = userEvent.setup();
		render(<CorpusMap source="alpha" ready />);
		await user.click(await screen.findByText(/2 documents, keep doc:\/\/a/));
		await waitFor(() => expect(asked.map((a) => a.generation)).toEqual(["g1"]));

		await user.click(screen.getByRole("button", { name: "Compute again" }));
		await waitFor(() =>
			expect(asked.map((a) => a.generation)).toEqual(["g1", "g2"]),
		);
		expect(screen.getAllByText("Document a").length).toBeGreaterThan(0);

		asked[0].answer([1, 0.111, 0.222]);
		asked[1].answer([1, 0.987, 0.654]);
		await screen.findByText("0.987");
		expect(screen.queryByText("0.111")).toBeNull();
	});

	it("drops a selection the new map no longer holds", async () => {
		const user = userEvent.setup();
		render(<CorpusMap source="alpha" ready />);
		await user.click(await screen.findByText(/2 documents, keep doc:\/\/a/));
		await screen.findByText("Document a");

		const without = finished("g2");
		without.result.points = without.result.points.filter((p) => p.id !== "a");
		without.result.families = [];
		let next = without;
		vi.mocked(fetch).mockImplementation(async (_url, init) =>
			init?.method === "POST"
				? Response.json({ ...next, status: "queued", result: null })
				: Response.json({ projection: next }),
		);
		await user.click(screen.getByRole("button", { name: "Compute again" }));
		await screen.findByText("Click a point.");

		next = finished("g3");
		await user.click(screen.getByRole("button", { name: "Compute again" }));
		await screen.findByText(/2 documents, keep doc:\/\/a/);
		expect(screen.getByText("Click a point.")).toBeTruthy();
	});

	it("never draws one map's scores against another map's points", async () => {
		vi.useFakeTimers({ shouldAdvanceTime: true });
		try {
			const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
			const { container } = render(<CorpusMap source="alpha" ready />);
			await user.click(await screen.findByText(/2 documents, keep doc:\/\/a/));
			await waitFor(() => expect(asked).toHaveLength(1));
			asked[0].answer([1, 0.111, 0.222]);
			await screen.findByText("0.222");

			// Another tab computes the map again: the same documents in another
			// order, read by the 30 s re-read. The first map's scores by position
			// would give the selected document its own 1.000.
			const reordered = finished("g2");
			reordered.result.points = [...reordered.result.points].reverse();
			current = reordered;
			const seen: string[] = [];
			// Record what each change added: a render undone within the same task
			// leaves the page as it was, but its nodes keep their text.
			const observer = new MutationObserver((records) => {
				for (const record of records) {
					for (const node of record.addedNodes)
						seen.push(node.textContent ?? "");
					if (record.type === "characterData")
						seen.push(record.target.textContent ?? "");
				}
			});
			observer.observe(container, {
				subtree: true,
				childList: true,
				characterData: true,
			});
			await vi.advanceTimersByTimeAsync(30_000);
			await waitFor(() =>
				expect(asked.map((a) => a.generation)).toEqual(["g1", "g2"]),
			);
			observer.disconnect();
			expect(seen.some((text) => text.includes("1.000"))).toBe(false);
		} finally {
			vi.useRealTimers();
		}
	});

	describe("on the canvas", () => {
		const SIZE = { width: 400, height: 300 };
		let restore: (() => void)[] = [];

		beforeEach(() => {
			for (const [key, value] of [
				["clientWidth", SIZE.width],
				["clientHeight", SIZE.height],
			] as const) {
				const previous = Object.getOwnPropertyDescriptor(
					HTMLElement.prototype,
					key,
				);
				Object.defineProperty(HTMLElement.prototype, key, {
					configurable: true,
					get: () => value,
				});
				restore.push(() => {
					if (previous)
						Object.defineProperty(HTMLElement.prototype, key, previous);
				});
			}
			HTMLCanvasElement.prototype.setPointerCapture = () => {};
		});

		afterEach(() => {
			for (const undo of restore) undo();
			restore = [];
		});

		// Where a point is drawn once the map is fitted to the canvas.
		const at = (id: string) => {
			const points = finished("g1").result.points;
			const p = points.find((q) => q.id === id) as (typeof points)[number];
			return toScreen(p, fitView(points, SIZE.width, SIZE.height));
		};
		const canvas = async () =>
			screen.findByRole("img", { name: /Map of 3 documents/ });
		const press = (element: Element, [x, y]: number[], moveBy = 0) => {
			fireEvent.pointerDown(element, { clientX: x, clientY: y, pointerId: 1 });
			if (moveBy)
				fireEvent.pointerMove(element, {
					clientX: x + moveBy,
					clientY: y,
					pointerId: 1,
				});
			fireEvent.pointerUp(element, {
				clientX: x + moveBy,
				clientY: y,
				pointerId: 1,
			});
		};

		it("draws every point, again on resize and theme change, and lets go on unmount", async () => {
			const calls: string[] = [];
			const context = new Proxy(
				{},
				{
					get:
						(_target, name) =>
						(...args: unknown[]) => {
							calls.push(String(name));
							return args;
						},
					set: () => true,
				},
			);
			HTMLCanvasElement.prototype.getContext = (() =>
				context) as unknown as typeof HTMLCanvasElement.prototype.getContext;
			const themes: (() => void)[] = [];
			const removed: (() => void)[] = [];
			vi.stubGlobal("matchMedia", () => ({
				addEventListener: (_: string, listener: () => void) =>
					themes.push(listener),
				removeEventListener: (_: string, listener: () => void) =>
					removed.push(listener),
			}));
			const resizes: (() => void)[] = [];
			let disconnected = 0;
			vi.stubGlobal(
				"ResizeObserver",
				class {
					constructor(callback: () => void) {
						resizes.push(callback);
					}
					observe() {}
					disconnect() {
						disconnected++;
					}
				},
			);
			const { unmount } = render(<CorpusMap source="alpha" ready />);
			await canvas();
			const arcs = () => calls.filter((name) => name === "arc").length;
			await waitFor(() => expect(arcs()).toBeGreaterThanOrEqual(3));

			let before = arcs();
			act(() => resizes.at(-1)?.());
			expect(arcs() - before).toBe(3);
			before = arcs();
			act(() => themes.at(-1)?.());
			expect(arcs() - before).toBe(3);

			const live = themes.at(-1);
			const connected = disconnected;
			unmount();
			expect(removed).toContain(live);
			expect(disconnected).toBe(connected + 1);
		});

		it("keeps the pan through a re-read, and refits for a new map", async () => {
			vi.useFakeTimers({ shouldAdvanceTime: true });
			try {
				render(<CorpusMap source="alpha" ready />);
				const map = await canvas();
				const [bx, by] = at("b");
				press(map, [bx, by], 40);

				// The same map read again 30 s later: the view stays panned.
				const reads = () =>
					vi
						.mocked(fetch)
						.mock.calls.filter(([url]) => String(url).endsWith("/projection"))
						.length;
				await vi.advanceTimersByTimeAsync(30_000);
				await waitFor(() => expect(reads()).toBe(2));
				press(map, [bx + 40, by]);
				await screen.findByText("Document b");
				press(map, [5, 5]);
				await screen.findByText("Click a point.");

				// A new map: the view refits, so b is back where fitting puts it.
				current = finished("g2");
				await vi.advanceTimersByTimeAsync(30_000);
				await waitFor(() => expect(reads()).toBe(3));
				press(map, [bx, by]);
				await screen.findByText("Document b");
			} finally {
				vi.useRealTimers();
			}
		});

		it("selects the point clicked, and clears on empty space", async () => {
			render(<CorpusMap source="alpha" ready />);
			const map = await canvas();
			press(map, at("b"));
			await screen.findByText("Document b");
			press(map, [5, 5]);
			await screen.findByText("Click a point.");
		});

		it("pans on a drag without selecting", async () => {
			render(<CorpusMap source="alpha" ready />);
			const map = await canvas();
			press(map, at("b"), 40);
			expect(screen.getByText("Click a point.")).toBeTruthy();
			const [x, y] = at("b");
			press(map, [x + 40, y]);
			await screen.findByText("Document b");
		});

		it("names the point under the pointer", async () => {
			render(<CorpusMap source="alpha" ready />);
			const map = await canvas();
			const [x, y] = at("c");
			fireEvent.pointerMove(map, { clientX: x, clientY: y });
			await screen.findByText("Document c · 3 chunks");
			fireEvent.pointerLeave(map);
			expect(screen.queryByText("Document c · 3 chunks")).toBeNull();
		});

		it("zooms about the pointer without scrolling the page", async () => {
			render(<CorpusMap source="alpha" ready />);
			const map = await canvas();
			const [ax, ay] = at("a");
			const wheel = new WheelEvent("wheel", {
				deltaY: -500,
				clientX: ax,
				clientY: ay,
				cancelable: true,
			});
			act(() => {
				map.dispatchEvent(wheel);
			});
			expect(wheel.defaultPrevented).toBe(true);
			press(map, [ax, ay]);
			await screen.findByText("Document a");
			const [bx, by] = at("b");
			fireEvent.pointerMove(map, { clientX: bx, clientY: by });
			expect(screen.queryByText("Document b · 3 chunks")).toBeNull();
		});
	});
});
