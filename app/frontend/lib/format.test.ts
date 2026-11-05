import { describe, expect, it } from "vitest";
import { readableUri } from "./format";

describe("readableUri", () => {
	it("decodes percent-encoded characters", () => {
		expect(
			readableUri("file:///data/sources/fek/%CE%91-58-2024.075234623b.pdf"),
		).toBe("file:///data/sources/fek/Α-58-2024.075234623b.pdf");
	});

	it("keeps a URI it cannot decode", () => {
		expect(readableUri("file:///data/100%.pdf")).toBe("file:///data/100%.pdf");
	});
});
