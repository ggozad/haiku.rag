import type { DatabaseEntry } from "./databases";

export interface Item {
	self_ref: string;
	label: string;
	text: string;
	page_numbers: number[];
	heading_level: number;
	// The chunks holding the item, in document order.
	chunk_ids: string[];
}

export interface OutlineNode {
	self_ref: string;
	level: number;
	title: string;
	page_numbers: number[];
	// Indexes into the position-ordered items: [first, last exclusive].
	item_range: [number, number];
	chunk_ids: string[];
	children: OutlineNode[];
}

export interface Block {
	kind: "heading" | "text" | "list" | "caption" | "table" | "picture" | "code";
	level: number;
	// The chunk an item opens and is coloured by: the first of its chunks.
	chunkId: string | null;
	// Alternates along the document, so neighbouring chunks are told apart.
	tint: 0 | 1 | null;
}

const KINDS: Record<string, Block["kind"]> = {
	section_header: "heading",
	title: "heading",
	list_item: "list",
	caption: "caption",
	table: "table",
	picture: "picture",
	chart: "picture",
	code: "code",
	formula: "code",
};

// Maps items to blocks. An item held by several chunks takes the first one's
// colour; an item held by none stays uncoloured.
export function blocksOf(items: Item[]): Block[] {
	const tints = new Map<string, 0 | 1>();
	return items.map((item) => {
		const chunkId = item.chunk_ids[0] ?? null;
		if (chunkId !== null && !tints.has(chunkId)) {
			tints.set(chunkId, (tints.size % 2) as 0 | 1);
		}
		return {
			kind: KINDS[item.label] ?? "text",
			level: item.label === "title" ? 1 : Math.max(1, item.heading_level),
			chunkId,
			tint: chunkId === null ? null : (tints.get(chunkId) ?? null),
		};
	});
}

// The number of distinct chunks under an outline node.
export function chunkCount(node: OutlineNode): number {
	return new Set(node.chunk_ids).size;
}

export function firstLine(text: string, length = 120): string {
	const line = text.split("\n").find((l) => l.trim()) ?? "";
	return line.length > length ? `${line.slice(0, length - 1)}…` : line;
}

// The databases a search covers: every available one the user has not unticked,
// so one that becomes available later is searched by default.
export function searched(
	databases: DatabaseEntry[],
	unticked: Set<string>,
): string[] {
	return databases.map((db) => db.name).filter((name) => !unticked.has(name));
}
