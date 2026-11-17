import { describe, expect, it } from "vitest";
import { formatBytes, formatTimeRange, publicUrl, readableUri } from "./format";

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

describe("formatBytes", () => {
	it("formats bytes in binary units", () => {
		expect(formatBytes(512)).toBe("512 B");
		expect(formatBytes(4096000)).toBe("3.9 MB");
		expect(formatBytes(5 * 1024 ** 5)).toBe("5120.0 TB");
	});
});

describe("formatTimeRange", () => {
	it("shows a span as two clock times", () => {
		expect(formatTimeRange([4.48, 3725.6])).toBe("00:00:04-01:02:05");
	});

	it("shows one time when both ends fall in the same second", () => {
		expect(formatTimeRange([10.0, 10.001])).toBe("00:00:10");
	});

	it("shows nothing without a range", () => {
		expect(formatTimeRange(null)).toBeNull();
		expect(formatTimeRange(undefined)).toBeNull();
	});
});
