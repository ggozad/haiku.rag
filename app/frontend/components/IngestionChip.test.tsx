/** @vitest-environment jsdom */
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import IngestionChip from "./IngestionChip";

let answer: () => Response;
let reads: number;

beforeEach(() => {
	reads = 0;
	Object.defineProperty(document, "hidden", {
		configurable: true,
		writable: true,
		value: false,
	});
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) => {
			expect(url).toBe("/api/ingester/health");
			reads++;
			return answer();
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

const health = (dead: number) =>
	Response.json({
		status: "ok",
		queue_counts: { queued: 0, claimed: 0, dead },
	});

describe("IngestionChip", () => {
	it("links to the ingestion page with the ingester's state", async () => {
		answer = () => health(2);
		render(<IngestionChip />);
		const chip = await screen.findByRole("link", { name: "2 dead letters" });
		expect(chip.getAttribute("href")).toBe("/ingestion");
	});

	it("says when the ingester does not answer", async () => {
		answer = () => Response.json({ detail: "down" }, { status: 502 });
		render(<IngestionChip />);
		await screen.findByRole("link", { name: "Ingester unreachable" });
	});

	it("reads again every 10 s while the page is visible", async () => {
		vi.useFakeTimers({ shouldAdvanceTime: true });
		answer = () => health(0);
		render(<IngestionChip />);
		await screen.findByText("Ingester idle");
		answer = () => health(1);
		await act(() => vi.advanceTimersByTimeAsync(10000));
		await screen.findByText("1 dead letter");
		expect(reads).toBe(2);
		(document as unknown as { hidden: boolean }).hidden = true;
		await act(() => vi.advanceTimersByTimeAsync(30000));
		expect(reads).toBe(2);
	});
});
