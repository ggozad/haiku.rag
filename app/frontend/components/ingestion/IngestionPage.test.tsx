/** @vitest-environment jsdom */
import {
	act,
	cleanup,
	render,
	screen,
	waitFor,
	within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import IngestionPage from "./IngestionPage";

const job = (id: string, status: string, extra = {}) => ({
	id,
	source_id: "docs",
	uri: `file:///docs/${id}.md`,
	op: "upsert",
	content_hash: null,
	revision: null,
	status,
	attempts: 1,
	max_attempts: 3,
	last_error: status === "dead" ? "conversion failed" : null,
	conversion_stalled: false,
	extra: null,
	enqueued_at: "2026-10-20T10:00:00Z",
	scheduled_at: "2026-10-20T10:00:00Z",
	claimed_at: "2026-10-20T10:00:01Z",
	claimed_by: "w-1",
	last_heartbeat_at: null,
	completed_at: status === "succeeded" ? "2026-10-20T10:00:05Z" : null,
	...extra,
});

const answers: Record<string, unknown> = {
	health: {
		status: "ok",
		queue_counts: { queued: 4, claimed: 1, succeeded: 12, dead: 1 },
		worker_count: 2,
		poller_count: 1,
		workers_alive: 2,
		pollers_alive: 1,
		worker_breaker_open: false,
		worker_breaker_consecutive_failures: 0,
	},
	sources: [
		{
			source_id: "docs",
			database: "docs",
			type: "filesystem",
			last_polled_at: null,
			circuit_breaker_open: false,
			last_skip_reason: null,
			last_error: null,
			last_error_at: null,
		},
	],
	providers: { docling_serve: [] },
	stats: {
		throughput: { succeeded_5m: 2, succeeded_30m: 5, succeeded_1h: 9 },
		workers: { busy: 1, total: 2 },
		oldest_queued_age_s: 90,
		dlq_by_source: { docs: 1 },
		queue_depth_by_source: { docs: 4 },
	},
	"jobs?status=claimed&limit=20": [job("running", "claimed")],
	"jobs?status=dead&limit=10": [job("broken", "dead")],
	"jobs?status=succeeded&limit=10": [job("done", "succeeded")],
	"databases/docs": { path: "/data/docs.lancedb" },
	"databases/papers": { path: "/data/papers.lancedb" },
};

let calls: { path: string; method: string }[];
let page: EventTarget & { hidden: boolean };

beforeEach(() => {
	calls = [];
	page = document as unknown as EventTarget & { hidden: boolean };
	Object.defineProperty(document, "hidden", {
		configurable: true,
		writable: true,
		value: false,
	});
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init?: RequestInit) => {
			const path = String(url).replace("/api/ingester/", "");
			const method = init?.method ?? "GET";
			calls.push({ path, method });
			if (method !== "GET")
				return Response.json(
					path.endsWith("/refresh") ? { refreshed: true } : {},
				);
			if (path.startsWith("jobs?limit=")) return Response.json([]);
			const answer = answers[path];
			return answer instanceof Response
				? answer.clone()
				: Response.json(answer ?? null);
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

const writes = () => calls.filter((call) => call.method !== "GET");
const browserReads = () =>
	calls.filter((call) => call.path.startsWith("jobs?limit=")).length;

describe("IngestionPage", () => {
	it("shows the queue, sources and jobs from one poll", async () => {
		render(<IngestionPage />);
		await screen.findByText("Writes docs");
		expect(screen.getByText("2/2 workers · 1/1 pollers")).toBeTruthy();
		expect(screen.getByText("2 / 9")).toBeTruthy();
		expect(screen.getByText("1/2")).toBeTruthy();
		expect(screen.getByText("filesystem")).toBeTruthy();
		expect(screen.getByText("file:///docs/running.md")).toBeTruthy();
		expect(screen.getByText("conversion failed")).toBeTruthy();
		expect(screen.getByText("file:///docs/done.md")).toBeTruthy();
		expect(screen.getByText("4s")).toBeTruthy();
	});

	it("names each source's database and its failed discovery", async () => {
		const [docs] = answers.sources as Record<string, unknown>[];
		answers.sources = [
			docs,
			{
				...docs,
				source_id: "papers-s3",
				database: "papers",
				type: "s3",
				last_error: "bucket unreadable",
				last_error_at: "2026-10-20T10:00:00Z",
			},
		];
		try {
			render(<IngestionPage />);
			await screen.findByText("Writes docs, papers");
			expect(
				screen.getByRole("columnheader", { name: "Database" }),
			).toBeTruthy();
			const row = screen.getByRole("row", { name: /papers-s3/ });
			expect(within(row).getByText("papers")).toBeTruthy();
			expect(within(row).getByText("discovery failed")).toBeTruthy();
			expect(within(row).getByText("bucket unreadable")).toBeTruthy();
		} finally {
			answers.sources = [docs];
		}
	});

	it("leaves out the database column when one database is ingested", async () => {
		render(<IngestionPage />);
		await screen.findByText("filesystem");
		expect(screen.queryByRole("columnheader", { name: "Database" })).toBeNull();
		expect(screen.queryByText("discovery failed")).toBeNull();
	});

	it("reads a database's info when its panel opens", async () => {
		const user = userEvent.setup();
		render(<IngestionPage />);
		await screen.findByText("Writes docs");
		await user.click(screen.getByRole("button", { name: "Database docs" }));
		await screen.findByText(/\/data\/docs\.lancedb/);
		expect(calls.some((call) => call.path === "databases/docs")).toBe(true);
	});

	it("keeps an open database panel through a failed sources poll", async () => {
		const user = userEvent.setup();
		render(<IngestionPage />);
		await screen.findByText("Writes docs");
		await user.click(screen.getByRole("button", { name: "Database docs" }));
		await screen.findByText(/\/data\/docs\.lancedb/);
		const sources = answers.sources;
		answers.sources = new Response("down", { status: 502 });
		try {
			await user.click(screen.getByRole("button", { name: "Refresh" }));
			await screen.findByText("The ingester answered 502.");
			expect(screen.getByText(/\/data\/docs\.lancedb/)).toBeTruthy();
		} finally {
			answers.sources = sources;
		}
	});

	it("refreshes a source, says so, and reads everything again", async () => {
		const user = userEvent.setup();
		render(<IngestionPage />);
		await screen.findByText("filesystem");
		const polls = calls.filter((call) => call.path === "health").length;
		const browsed = browserReads();
		await user.click(screen.getByRole("button", { name: "Refresh" }));
		await screen.findByText("Refreshed docs.");
		expect(writes()).toEqual([
			{ path: "sources/docs/refresh", method: "POST" },
		]);
		await waitFor(() =>
			expect(calls.filter((call) => call.path === "health").length).toBe(
				polls + 1,
			),
		);
		await waitFor(() => expect(browserReads()).toBe(browsed + 1));
	});

	it("retries a dead letter", async () => {
		const user = userEvent.setup();
		render(<IngestionPage />);
		await screen.findByText("conversion failed");
		await user.click(screen.getByRole("button", { name: "Retry" }));
		await screen.findByText("Queued file:///docs/broken.md again.");
		expect(writes()).toEqual([{ path: "jobs/broken/retry", method: "POST" }]);
	});

	it("cancels a job only once confirmed", async () => {
		const user = userEvent.setup();
		render(<IngestionPage />);
		await screen.findByText("file:///docs/running.md");
		await user.click(screen.getByRole("button", { name: "Cancel" }));
		let dialog = await screen.findByRole("dialog");
		await user.click(within(dialog).getByRole("button", { name: "Keep it" }));
		expect(writes()).toEqual([]);
		await user.click(screen.getByRole("button", { name: "Cancel" }));
		dialog = await screen.findByRole("dialog");
		await user.click(
			within(dialog).getByRole("button", { name: "Cancel the job" }),
		);
		await screen.findByText("Cancelled file:///docs/running.md.");
		expect(writes()).toEqual([{ path: "jobs/running", method: "DELETE" }]);
	});

	it("polls every 3 s while visible, and pauses while hidden", async () => {
		vi.useFakeTimers({ shouldAdvanceTime: true });
		render(<IngestionPage />);
		await screen.findByText("filesystem");
		const polls = () => calls.filter((call) => call.path === "health").length;
		await act(() => vi.advanceTimersByTimeAsync(3000));
		expect(polls()).toBe(2);
		page.hidden = true;
		act(() => {
			document.dispatchEvent(new Event("visibilitychange"));
		});
		await act(() => vi.advanceTimersByTimeAsync(9000));
		expect(polls()).toBe(2);
		page.hidden = false;
		act(() => {
			document.dispatchEvent(new Event("visibilitychange"));
		});
		await act(() => vi.advanceTimersByTimeAsync(0));
		expect(polls()).toBe(3);
	});
});
