/** @vitest-environment jsdom */
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import JobsBrowser from "./JobsBrowser";
import JobTable from "./JobTable";
import type { Job } from "./types";

const job = (id: string, uri: string, extra: Partial<Job> = {}): Job => ({
	id,
	source_id: "docs",
	uri,
	op: "upsert",
	content_hash: null,
	revision: null,
	status: "claimed",
	attempts: 1,
	max_attempts: 3,
	last_error: null,
	conversion_stalled: false,
	extra: null,
	enqueued_at: "2026-10-20T10:00:00Z",
	scheduled_at: "2026-10-20T10:00:00Z",
	claimed_at: new Date(Date.now() - 125_000).toISOString(),
	claimed_by: "1234-abcd-0",
	last_heartbeat_at: null,
	completed_at: null,
	...extra,
});

const jobs = [
	job("a", "s3://bucket/reports/Q3%20summary.pdf"),
	job("b", "file:///docs/gone.md", { op: "delete" }),
];

const headings = () =>
	screen.getAllByRole("columnheader").map((header) => header.textContent);

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("JobTable", () => {
	it("shows each job's file name above its full URI", () => {
		render(
			<JobTable jobs={jobs} columns={["file"]} onOpen={() => {}} empty="" />,
		);
		expect(screen.getByText("Q3 summary.pdf")).toBeTruthy();
		expect(screen.getByText("s3://bucket/reports/Q3 summary.pdf")).toBeTruthy();
		expect(headings()).toContain("File");
	});

	it("marks delete jobs and nothing else", () => {
		render(
			<JobTable jobs={jobs} columns={["file"]} onOpen={() => {}} empty="" />,
		);
		const gone = screen.getByText("gone.md").closest("tr") as HTMLElement;
		const kept = screen
			.getByText("Q3 summary.pdf")
			.closest("tr") as HTMLElement;
		expect(within(gone).getByText("delete")).toBeTruthy();
		expect(within(kept).queryByText("delete")).toBeNull();
		expect(within(kept).queryByText("upsert")).toBeNull();
	});

	it("shows how long an active job has been running", () => {
		render(
			<JobTable
				jobs={jobs}
				columns={["file", "running"]}
				onOpen={() => {}}
				empty=""
			/>,
		);
		expect(headings()).toContain("Running");
		const row = screen.getByText("Q3 summary.pdf").closest("tr");
		expect(row?.textContent).toContain("2m");
	});

	it("copies the full URI without opening the job", async () => {
		const user = userEvent.setup();
		const writeText = vi
			.spyOn(navigator.clipboard, "writeText")
			.mockResolvedValue(undefined);
		const onOpen = vi.fn();
		render(
			<JobTable jobs={jobs} columns={["file"]} onOpen={onOpen} empty="" />,
		);
		const row = screen.getByText("Q3 summary.pdf").closest("tr") as HTMLElement;
		await user.click(within(row).getByRole("button", { name: "Copy URI" }));
		expect(writeText).toHaveBeenCalledTimes(1);
		expect(writeText).toHaveBeenCalledWith(
			"s3://bucket/reports/Q3%20summary.pdf",
		);
		expect(onOpen).not.toHaveBeenCalled();
		await user.click(screen.getByText("Q3 summary.pdf"));
		expect(onOpen).toHaveBeenCalledWith(jobs[0]);
	});
});

describe("JobTable copy failure", () => {
	const copy = async () => {
		render(
			<JobTable jobs={jobs} columns={["file"]} onOpen={() => {}} empty="" />,
		);
		const row = screen.getByText("Q3 summary.pdf").closest("tr") as HTMLElement;
		const button = within(row).getByRole("button", { name: "Copy URI" });
		await userEvent.click(button);
		return button;
	};

	it("names the browser's reason for a refused copy", async () => {
		userEvent.setup();
		vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
			new DOMException("denied", "NotAllowedError"),
		);
		expect((await copy()).title).toBe("Copy failed: denied");
	});

	it("says copying needs a secure page without a clipboard", async () => {
		userEvent.setup();
		vi.stubGlobal("navigator", { ...navigator, clipboard: undefined });
		expect((await copy()).title).toBe("Copying needs HTTPS or localhost");
	});
});

describe("JobsBrowser", () => {
	it("shows file names, a delete badge and no op column", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => Response.json(jobs)),
		);
		render(<JobsBrowser sources={["docs"]} revision={0} />);
		await screen.findByText("Q3 summary.pdf");
		expect(headings()).not.toContain("Op");
		expect(headings()).toContain("File");
		const gone = screen.getByText("gone.md").closest("tr") as HTMLElement;
		expect(within(gone).getByText("delete")).toBeTruthy();
	});
});
