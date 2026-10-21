/** @vitest-environment jsdom */
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useJson } from "./useJson";

afterEach(() => vi.unstubAllGlobals());

describe("useJson", () => {
	it("reads the url again when its refresh key changes, keeping what it shows", async () => {
		let answer = 1;
		const fetch = vi.fn(async () => Response.json({ n: answer }));
		vi.stubGlobal("fetch", fetch);
		const { result, rerender } = renderHook(
			({ refresh }) => useJson<{ n: number }>("/x", refresh),
			{ initialProps: { refresh: 10 } },
		);
		await waitFor(() => expect(result.current.data).toEqual({ n: 1 }));
		answer = 2;
		rerender({ refresh: 11 });
		expect(result.current.data).toEqual({ n: 1 });
		await waitFor(() => expect(result.current.data).toEqual({ n: 2 }));
		expect(fetch).toHaveBeenCalledTimes(2);
	});

	it("clears what it shows when the url changes", async () => {
		vi.stubGlobal("fetch", async (url: string) => Response.json({ url }));
		const { result, rerender } = renderHook(({ url }) => useJson(url), {
			initialProps: { url: "/a" },
		});
		await waitFor(() => expect(result.current.data).toEqual({ url: "/a" }));
		rerender({ url: "/b" });
		expect(result.current.data).toBe(null);
		await waitFor(() => expect(result.current.data).toEqual({ url: "/b" }));
	});
});
