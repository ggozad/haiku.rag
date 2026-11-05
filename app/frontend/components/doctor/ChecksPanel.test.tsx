/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Check } from "@/lib/doctor";
import ChecksPanel from "./ChecksPanel";

const check = (name: string, severity: Check["severity"]): Check => ({
	name,
	severity,
	message: `${name} message`,
	remediation: null,
	details: [],
});

function serve(results: Check[]) {
	vi.stubGlobal(
		"fetch",
		vi.fn(async () =>
			Response.json({
				run: {
					kind: "run",
					status: "done",
					progress: [],
					queued_at: "t",
					started_at: "t",
					finished_at: "2026-11-05T10:00:00Z",
					result: { results },
					error: null,
				},
			}),
		),
	);
}

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("ChecksPanel", () => {
	it("leads with the problems and folds the passing checks", async () => {
		serve([
			check("tables_present", "ok"),
			check("fts_index_coverage", "warn"),
			check("vector_dimension", "ok"),
		]);
		render(<ChecksPanel source="alpha" ready />);
		await screen.findByText("1 warning");
		expect(screen.getByText("fts_index_coverage message")).toBeTruthy();
		expect(screen.queryByText("tables_present message")).toBeNull();
		expect(screen.queryByText(/failed/)).toBeNull();

		await userEvent.setup().click(screen.getByText("Passing checks (2)"));
		expect(screen.getByText("tables_present message")).toBeTruthy();
	});

	it("says when every check passes", async () => {
		serve([check("tables_present", "ok"), check("vector_dimension", "ok")]);
		render(<ChecksPanel source="alpha" ready />);
		expect(await screen.findByText("All 2 checks pass")).toBeTruthy();
	});
});
