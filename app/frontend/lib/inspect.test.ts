import { describe, expect, it } from "vitest";
import { blocksOf, chunkCount, firstLine, type Item } from "./inspect";

const item = (label: string, chunk_ids: string[], heading_level = 0): Item => ({
	self_ref: `#/x/${Math.random()}`,
	label,
	text: "t",
	page_numbers: [],
	heading_level,
	chunk_ids,
});

describe("blocksOf", () => {
	it("colours items by their first chunk, alternating along the document", () => {
		const blocks = blocksOf([
			item("section_header", ["a"], 2),
			item("text", ["a", "b"]),
			item("text", ["b"]),
			item("text", []),
			item("list_item", ["c"]),
			item("picture", ["a"]),
			item("title", ["a"]),
		]);
		expect(blocks.map((b) => [b.kind, b.chunkId, b.tint])).toEqual([
			["heading", "a", 0],
			["text", "a", 0],
			["text", "b", 1],
			["text", null, null],
			["list", "c", 0],
			["picture", "a", 0],
			["heading", "a", 0],
		]);
		expect(blocks[0].level).toBe(2);
		expect(blocks[6].level).toBe(1);
	});
});

describe("outline and text", () => {
	it("counts distinct chunks and cuts a first line", () => {
		expect(
			chunkCount({
				self_ref: "#/texts/0",
				level: 1,
				title: "t",
				page_numbers: [],
				item_range: [0, 2],
				chunk_ids: ["a", "b", "a"],
				children: [],
			}),
		).toBe(2);
		expect(firstLine("\n  \nhello\nworld")).toBe("hello");
		expect(firstLine("abcdef", 4)).toBe("abc…");
	});
});
