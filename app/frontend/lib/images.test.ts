import { describe, expect, it } from "vitest";
import { fitWithin, imagesOf, userContent } from "./images";

describe("images", () => {
	it("shrinks the long edge to the limit and keeps the ratio", () => {
		expect(fitWithin(4096, 1024)).toEqual({ width: 2048, height: 512 });
		expect(fitWithin(1000, 3000, 300)).toEqual({ width: 100, height: 300 });
		expect(fitWithin(640, 480)).toEqual({ width: 640, height: 480 });
	});

	it("builds the user content from text and images, and reads it back", () => {
		const image = { data: "AAAA", mimeType: "image/png", name: "a.png" };
		expect(userContent("plain", [])).toBe("plain");
		const content = userContent("look", [image]);
		expect(content).toEqual([
			{ type: "text", text: "look" },
			{
				type: "image",
				source: { type: "data", value: "AAAA", mimeType: "image/png" },
			},
		]);
		expect(userContent("", [image])).toHaveLength(1);
		expect(imagesOf(content)).toEqual(["data:image/png;base64,AAAA"]);
		expect(imagesOf("text")).toEqual([]);
	});
});
