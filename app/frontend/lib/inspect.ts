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

// Chunk text as one line of prose: markdown markers and table rules dropped.
export function plainText(markdown: string): string {
	return markdown
		.replace(/^\s*(#{1,6}|[-+](?=\s)|\|?\s*:?-{3,}.*)\s*/gm, "")
		.replace(/\*+|\|/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

// A chunk's item labels, each once with its count.
export function labelCounts(labels: string[]): string {
	const counts = new Map<string, number>();
	for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
	return [...counts]
		.map(([label, n]) => (n > 1 ? `${label} ×${n}` : label))
		.join(", ");
}

// Keys the ingester owns, and keys a document's header shows on their own.
const UNLISTED = new Set([
	"content_type",
	"md5",
	"source_revision",
	"source_id",
	"parent_uri",
	"title",
	"url",
]);

// The metadata a document's header shows inline: short scalar values, in
// their stored order. The rest stays under "All metadata".
export function headlineMetadata(
	meta: Record<string, unknown>,
): [string, string][] {
	const out: [string, string][] = [];
	for (const [key, value] of Object.entries(meta)) {
		if (UNLISTED.has(key) || value === null || typeof value === "object")
			continue;
		const text = String(value);
		if (text.length > 80) continue;
		out.push([key, text]);
		if (out.length === 8) break;
	}
	return out;
}

// The databases a search covers: every available one the user has not unticked,
// so one that becomes available later is searched by default.
export function searched(
	databases: DatabaseEntry[],
	unticked: Set<string>,
): string[] {
	return databases.map((db) => db.name).filter((name) => !unticked.has(name));
}

// One entry of the editor's metadata (`metadata.user`): a string value is
// edited as text, any other value as JSON.
export interface EditorRow {
	key: string;
	value: string;
	json: boolean;
}

// The rows of a stored `metadata.user`, none for a value that is not an object.
export function editorRows(user: unknown): EditorRow[] {
	if (user === null || typeof user !== "object" || Array.isArray(user))
		return [];
	return Object.entries(user).map(([key, value]) =>
		typeof value === "string"
			? { key, value, json: false }
			: { key, value: JSON.stringify(value), json: true },
	);
}

// The rows as the object the ingester stores, or why they are not one.
export function editorMetadata(
	rows: EditorRow[],
): Record<string, unknown> | string {
	const entries = new Map<string, unknown>();
	for (const row of rows) {
		const key = row.key;
		if (!key && !row.value) continue;
		if (!key.trim()) return "Every value needs a key.";
		if (entries.has(key)) return `${key} appears twice.`;
		if (!row.json) {
			entries.set(key, row.value);
			continue;
		}
		try {
			entries.set(key, JSON.parse(row.value));
		} catch {
			return `${key} is not valid JSON.`;
		}
	}
	return Object.fromEntries(entries);
}
