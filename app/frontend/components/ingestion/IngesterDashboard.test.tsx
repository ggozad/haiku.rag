/** @vitest-environment jsdom */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The ingester's own dashboard, a static page shipped in haiku.rag-slim.
// Vitest runs from app/frontend.
const page = readFileSync(
	resolve("../../haiku_rag_slim/haiku/rag/ingester/api/static/index.html"),
	"utf-8",
);

const job = (id: string, uri: string, extra = {}) => ({
	id: `${id}-0000-0000-0000-000000000000`,
	source_id: "docs",
	uri,
	op: "upsert",
	status: "claimed",
	attempts: 1,
	max_attempts: 3,
	last_error: null,
	enqueued_at: "2026-10-20T10:00:00Z",
	claimed_at: "2026-10-20T10:00:01Z",
	claimed_by: "1234-abcd-0",
	completed_at: null,
	...extra,
});

const source = (source_id: string, files = {}) => ({
	source_id,
	database: "docs",
	type: "fs",
	last_polled_at: null,
	circuit_breaker_open: false,
	last_skip_reason: null,
	last_error: null,
	last_error_at: null,
	unsupported_files: {},
	known: 0,
	ingested: 0,
	pending: 0,
	failed: 0,
	waiting: 0,
	...files,
});

function answers(): Record<string, unknown> {
	return {
		health: {
			status: "ok",
			queue_counts: { queued: 0, claimed: 2, succeeded: 2, dead: 1 },
			worker_count: 2,
			poller_count: 1,
			worker_breaker_open: false,
			worker_breaker_consecutive_failures: 0,
			failed_databases: [],
		},
		sources: [
			source("docs", {
				known: 8,
				ingested: 5,
				pending: 1,
				failed: 1,
				waiting: 1,
			}),
			source("empty"),
		],
		providers: { docling_serve: [] },
		stats: {
			throughput: { succeeded_5m: 0, succeeded_30m: 0, succeeded_1h: 0 },
			workers: { busy: 2, total: 2 },
			oldest_queued_age_s: null,
			dlq_by_source: {},
			rejected_by_source: {},
			queue_depth_by_source: {},
		},
		"jobs?status=claimed": [
			job("aaaaaaaa", "s3://bucket/reports/Q3%20summary.pdf", {
				claimed_at: new Date(Date.now() - 125_000).toISOString(),
			}),
			job("bbbbbbbb", "file:///docs/gone.md", { op: "delete" }),
		],
		"jobs?status=dead": [
			job("cccccccc", "https://example.com/handbook/", {
				status: "dead",
				last_error: "conversion failed",
				completed_at: "2026-10-20T10:00:05Z",
			}),
		],
		"jobs?status=succeeded": [
			job("dddddddd", "https://example.com/a/b.pdf?v=2#page=3", {
				status: "succeeded",
				completed_at: "2026-10-20T10:00:05Z",
			}),
		],
	};
}

// Document listeners added by the page's script, removed after each test.
const listeners: Parameters<Document["addEventListener"]>[] = [];

async function open() {
	const doc = new DOMParser().parseFromString(page, "text/html");
	document.body.innerHTML = doc.body.innerHTML;
	const responses = answers();
	vi.stubGlobal(
		"fetch",
		vi.fn(async (path: string) => {
			const key = Object.keys(responses).find(
				(k) => path === k || path.startsWith(`${k}&`),
			);
			return new Response(JSON.stringify(key ? responses[key] : {}));
		}),
	);
	const add = document.addEventListener.bind(document);
	vi.spyOn(document, "addEventListener").mockImplementation((...args) => {
		listeners.push(args);
		add(...args);
	});
	// The page's inline script: parsed markup never runs it.
	new Function(doc.querySelector("script")?.textContent ?? "")();
	await waitFor(() =>
		expect(
			document.getElementById("recent-body")?.querySelector("table"),
		).not.toBeNull(),
	);
}

const table = (id: string) =>
	within(document.getElementById(id) as HTMLElement);

const headings = (id: string) =>
	Array.from(
		document.getElementById(id)?.querySelectorAll("th") ?? [],
		(th) => th.textContent,
	);

describe("ingester dashboard job tables", () => {
	beforeEach(() => {
		vi.useFakeTimers({ toFake: ["setInterval"] });
	});

	afterEach(() => {
		vi.useRealTimers();
		vi.unstubAllGlobals();
		vi.restoreAllMocks();
		for (const [type, listener, options] of listeners.splice(0))
			document.removeEventListener(type, listener, options);
		document.body.innerHTML = "";
	});

	it("shows each job's file name above its full URI", async () => {
		await open();
		const active = table("active-body");
		expect(active.getByText("Q3 summary.pdf")).toBeTruthy();
		expect(
			active.getByText("s3://bucket/reports/Q3%20summary.pdf"),
		).toBeTruthy();
		expect(table("dead-body").getByText("handbook")).toBeTruthy();
		expect(table("recent-body").getByText("b.pdf")).toBeTruthy();
	});

	it("shows no ID, Op or Worker column", async () => {
		await open();
		for (const id of ["active-body", "dead-body", "recent-body"]) {
			const shown = headings(id);
			expect(shown).not.toContain("ID");
			expect(shown).not.toContain("Op");
			expect(shown).not.toContain("Worker");
		}
		expect(document.getElementById("active-body")?.textContent).not.toContain(
			"1234-abcd-0",
		);
	});

	it("marks delete jobs and nothing else", async () => {
		await open();
		const active = table("active-body");
		const gone = active.getByText("gone.md").closest("tr") as HTMLElement;
		const kept = active
			.getByText("Q3 summary.pdf")
			.closest("tr") as HTMLElement;
		expect(within(gone).getByText("DEL")).toBeTruthy();
		expect(kept.querySelector(".badge")).toBeNull();
		expect(
			document.getElementById("recent-body")?.querySelector(".badge"),
		).toBeNull();
	});

	it("shows how long an active job has been running", async () => {
		await open();
		expect(headings("active-body")).toContain("Running");
		const row = table("active-body").getByText("Q3 summary.pdf").closest("tr");
		expect(row?.textContent).toMatch(/2m [56]s(?! ago)/);
	});

	it("shows how much of each source is ingested", async () => {
		await open();
		const sources = table("sources-body");
		const docs = sources.getByText("docs").closest("tr") as HTMLElement;
		const widths = Array.from(
			docs.querySelectorAll<HTMLElement>(".bar > span"),
			(segment) => segment.style.width,
		);
		expect(widths).toEqual(["62.5%", "12.5%", "12.5%", "12.5%"]);
		expect(docs.textContent).toContain("62%");
		expect(docs.textContent).toMatch(
			/5 ingested.*1 pending.*1 failed.*1 waiting/,
		);
		const empty = sources.getByText("empty").closest("tr") as HTMLElement;
		expect(empty.querySelector(".bar")).toBeNull();
	});

	it("copies the full URI when the file cell is clicked", async () => {
		const writeText = vi.fn().mockResolvedValue(undefined);
		Object.defineProperty(navigator, "clipboard", {
			value: { writeText },
			configurable: true,
		});
		await open();
		await userEvent.click(table("dead-body").getByText("handbook"));
		expect(writeText).toHaveBeenCalledTimes(1);
		expect(writeText).toHaveBeenCalledWith("https://example.com/handbook/");
	});
});
