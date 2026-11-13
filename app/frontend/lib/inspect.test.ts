import { describe, expect, it } from "vitest";
import {
	blocksOf,
	chunkCount,
	editorMetadata,
	editorRows,
	firstLine,
	headlineMetadata,
	type Item,
	labelCounts,
	plainText,
	searched,
} from "./inspect";

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

describe("searched", () => {
	const db = (name: string) => ({ name, available: true, error: null });

	it("names every available database the user has not unticked", () => {
		expect(searched([db("alpha")], new Set())).toEqual(["alpha"]);
		expect(searched([db("alpha"), db("beta")], new Set(["beta"]))).toEqual([
			"alpha",
		]);
		expect(searched([db("alpha")], new Set(["alpha"]))).toEqual([]);
	});

	it("includes a database that becomes available later", () => {
		const unticked = new Set(["beta"]);
		expect(searched([db("alpha")], unticked)).toEqual(["alpha"]);
		expect(searched([db("alpha"), db("beta"), db("gamma")], unticked)).toEqual([
			"alpha",
			"gamma",
		]);
	});
});

describe("plainText", () => {
	it("drops markdown emphasis, headings and table pipes", () => {
		expect(plainText("## Άρθρο 4\n**β)** να *μην* έχει: *** α)")).toBe(
			"Άρθρο 4 β) να μην έχει: α)",
		);
		expect(plainText("| Rate | Food |\n|---|---|\n| 13% | yes |")).toBe(
			"Rate Food 13% yes",
		);
	});

	it("keeps list markers' text and underscores inside words", () => {
		expect(plainText("- first_item\n- second")).toBe("first_item second");
		expect(plainText("-5% change")).toBe("-5% change");
	});
});

describe("headlineMetadata", () => {
	it("keeps short scalar fields in order and leaves out what the header shows elsewhere", () => {
		expect(
			headlineMetadata({
				number: "Α.1077/2025",
				issued: "2025-05-29",
				pages: 12,
				title: "Shown as the heading",
				url: "https://example.org/a",
				md5: "81134a21",
				source_id: "diavgeia",
				content_type: "application/pdf",
				source_revision: "1793",
				parent_uri: null,
				tags: ["a", "b"],
				subject: "x".repeat(81),
			}),
		).toEqual([
			["number", "Α.1077/2025"],
			["issued", "2025-05-29"],
			["pages", "12"],
		]);
	});

	it("shows at most eight", () => {
		const meta = Object.fromEntries(
			Array.from({ length: 10 }, (_, n) => [`k${n}`, n]),
		);
		expect(headlineMetadata(meta)).toHaveLength(8);
	});
});

describe("labelCounts", () => {
	it("names each label once, in first-seen order, with its count", () => {
		expect(
			labelCounts(["text", "list_item", "list_item", "text", "list_item"]),
		).toBe("text ×2, list_item ×3");
		expect(labelCounts(["table"])).toBe("table");
		expect(labelCounts([])).toBe("");
	});
});

describe("editorRows", () => {
	it("edits a string as text and any other value as JSON", () => {
		expect(editorRows({ reviewer: "ana", tags: ["a"], n: 2 })).toEqual([
			{ key: "reviewer", value: "ana", json: false },
			{ key: "tags", value: '["a"]', json: true },
			{ key: "n", value: "2", json: true },
		]);
	});

	it("reads a value that is not an object as no rows", () => {
		expect(editorRows(undefined)).toEqual([]);
		expect(editorRows("legacy")).toEqual([]);
		expect(editorRows(["a"])).toEqual([]);
	});
});

describe("editorMetadata", () => {
	it("turns the rows back into the stored object", () => {
		const user = { reviewer: "ana", tags: ["a"], n: 2 };
		expect(editorMetadata(editorRows(user))).toEqual(user);
	});

	it("drops a blank row and keeps a key as written", () => {
		expect(
			editorMetadata([
				{ key: " team ", value: "x", json: false },
				{ key: "", value: "", json: false },
			]),
		).toEqual({ " team ": "x" });
	});

	it("keeps a key that names an Object property as data", () => {
		const out = editorMetadata([{ key: "__proto__", value: "x", json: false }]);
		expect(Object.getPrototypeOf(out)).toBe(Object.prototype);
		expect(Object.keys(out)).toEqual(["__proto__"]);
	});

	it.each([
		[[{ key: "", value: "x", json: false }], "Every value needs a key."],
		[[{ key: "  ", value: "x", json: false }], "Every value needs a key."],
		[
			[
				{ key: "a", value: "x", json: false },
				{ key: "a", value: "y", json: false },
			],
			"a appears twice.",
		],
		[[{ key: "a", value: "{", json: true }], "a is not valid JSON."],
	])("names why the rows are not an object", (rows, reason) => {
		expect(editorMetadata(rows)).toBe(reason);
	});
});
