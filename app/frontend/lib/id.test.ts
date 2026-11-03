import { afterEach, describe, expect, it, vi } from "vitest";
import { newId } from "./id";

const V4 =
	/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe("newId", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("uses randomUUID where the browser provides it", () => {
		vi.stubGlobal("crypto", { randomUUID: () => "from-random-uuid" });
		expect(newId()).toBe("from-random-uuid");
	});

	it("builds a v4 UUID outside a secure context", () => {
		const real = crypto;
		vi.stubGlobal("crypto", {
			getRandomValues: real.getRandomValues.bind(real),
		});
		const first = newId();
		expect(first).toMatch(V4);
		expect(newId()).not.toBe(first);
	});
});
