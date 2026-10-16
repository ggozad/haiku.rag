import { describe, expect, it } from "vitest";
import { contentText } from "./sessionStorage";

describe("contentText", () => {
	it("reads a string or the text parts of a list", () => {
		expect(contentText("plain")).toBe("plain");
		expect(
			contentText([
				{ type: "text", text: "a " },
				{ type: "image", source: { type: "data", value: "x" } },
				{ type: "text", text: "b" },
			]),
		).toBe("a b");
		expect(contentText(undefined)).toBe("");
	});
});
