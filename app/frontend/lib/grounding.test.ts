import { describe, expect, it } from "vitest";
import {
	boxStyle,
	originalUrl,
	pageBoxesUrl,
	pageImageUrl,
	withBoxes,
} from "./grounding";

describe("boxStyle", () => {
	it("places a bottom-left box as top-left percentages of the page", () => {
		expect(boxStyle({ l: 61.2, t: 712.8, r: 306, b: 633.6 }, 612, 792)).toEqual(
			{ left: "10%", top: "10%", width: "40%", height: "10%" },
		);
	});

	it("places a box on a page the original turns sideways", () => {
		expect(boxStyle({ l: 79.2, t: 550.8, r: 396, b: 367.2 }, 792, 612)).toEqual(
			{ left: "10%", top: "10%", width: "40%", height: "30%" },
		);
	});

	it("takes a box whatever order its edges come in", () => {
		expect(boxStyle({ l: 306, t: 633.6, r: 61.2, b: 712.8 }, 612, 792)).toEqual(
			{ left: "10%", top: "10%", width: "40%", height: "10%" },
		);
	});
});

describe("withBoxes", () => {
	it("draws the boxes of the pages it is given on the document's pages", () => {
		const box = { l: 1, t: 2, r: 3, b: 0, matched: true };
		expect(
			withBoxes(
				[
					{ page: 1, width: 10, height: 20 },
					{ page: 2, width: 10, height: 20 },
				],
				[{ page: 2, width: 10, height: 20, boxes: [box] }],
			),
		).toEqual([
			{ page: 1, width: 10, height: 20, boxes: [] },
			{ page: 2, width: 10, height: 20, boxes: [box] },
		]);
	});
});

describe("original URLs", () => {
	it("address the original and its pages by database, document and MD5", () => {
		expect(originalUrl("my db", "doc/1", "abc")).toBe(
			"/api/backend/api/inspect/my%20db/documents/doc%2F1/original?md5=abc",
		);
		expect(pageImageUrl("my db", "doc/1", 3, "abc")).toBe(
			"/api/backend/api/inspect/my%20db/documents/doc%2F1/pages/3?md5=abc",
		);
		expect(pageBoxesUrl("my db", "doc/1", 3)).toBe(
			"/api/backend/api/inspect/my%20db/documents/doc%2F1/pages/3/boxes",
		);
	});
});
