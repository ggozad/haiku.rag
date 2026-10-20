/** @vitest-environment jsdom */
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useJob } from "./useJob";

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("useJob", () => {
	it("reports a start the network refused, and starts on a retry", async () => {
		const job = { kind: "run", status: "queued", progress: [], result: null };
		let refuse = true;
		let started: typeof job | null = null;
		vi.stubGlobal(
			"fetch",
			vi.fn(async (_url: string, init?: RequestInit) => {
				if (init?.method !== "POST") return Response.json({ run: started });
				if (refuse) {
					refuse = false;
					throw new TypeError("network down");
				}
				started = job;
				return Response.json(job);
			}),
		);
		const { result } = renderHook(() => useJob("/status", "run"));
		// The page loads before anyone presses Run.
		await act(async () => {});
		await act(() => result.current.start("/start"));
		expect(result.current.error).toBe("network down");
		await act(() => result.current.start("/start"));
		expect(result.current.error).toBeNull();
		expect(result.current.job?.status).toBe("queued");
	});
});
