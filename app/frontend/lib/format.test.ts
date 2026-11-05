import { describe, expect, it } from "vitest";
import { publicUrl, readableUri } from "./format";

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

describe("publicUrl", () => {
	it("returns the document's http link", () => {
		expect(publicUrl({ url: "https://diavgeia.gov.gr/doc/1" })).toBe(
			"https://diavgeia.gov.gr/doc/1",
		);
	});

	it("ignores a missing, non-web or non-string url", () => {
		expect(publicUrl(undefined)).toBeNull();
		expect(publicUrl({})).toBeNull();
		expect(publicUrl({ url: "file:///data/a.pdf" })).toBeNull();
		expect(publicUrl({ url: "javascript:alert(1)" })).toBeNull();
		expect(publicUrl({ url: 3 })).toBeNull();
	});
});
