/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import SourcesPanel from "./SourcesPanel";
import type { Source } from "./types";

const source = (source_id: string, files: Partial<Source> = {}): Source => ({
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

const row = (name: string) =>
	screen.getByText(name).closest("tr") as HTMLElement;

afterEach(cleanup);

describe("SourcesPanel", () => {
	it("shows how much of each source is ingested", () => {
		render(
			<SourcesPanel
				sources={[
					source("docs", {
						known: 8,
						ingested: 5,
						pending: 1,
						failed: 1,
						waiting: 1,
					}),
					source("empty"),
				]}
				stats={undefined}
				busy={null}
				onScan={() => {}}
				onRetry={() => {}}
			/>,
		);
		expect(
			screen.getAllByRole("columnheader").map((h) => h.textContent),
		).toContain("Files");
		const bar = screen.getByRole("img", { name: "62% of 8 files ingested" });
		expect(
			Array.from(bar.children, (segment) => [
				segment.getAttribute("title"),
				(segment as HTMLElement).style.width,
			]),
		).toEqual([
			["5 ingested", "62.5%"],
			["1 pending", "12.5%"],
			["1 failed", "12.5%"],
			["1 waiting", "12.5%"],
		]);
		expect(row("docs").textContent).toMatch(
			/5 ingested.*1 pending.*1 failed.*1 waiting/,
		);
		expect(row("empty").querySelector('[role="img"]')).toBeNull();
	});
});
