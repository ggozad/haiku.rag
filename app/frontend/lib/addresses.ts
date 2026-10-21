// The inspector's and the doctor's addresses. Every view's state is in its URL.

export type InspectView =
	| { kind: "home" }
	| { kind: "search" }
	| { kind: "database"; db: string }
	| { kind: "document"; db: string; doc: string }
	| { kind: "chunk"; db: string; doc: string; chunk: string }
	| { kind: "unknown" };

// `path` is the `/inspect` route's catch-all segments, already decoded. Search
// has its own address, `/search`.
export function inspectView(path: string[] | undefined): InspectView {
	const [db, doc, part, chunk, ...rest] = path ?? [];
	if (db === undefined) return { kind: "home" };
	if (doc === undefined) return { kind: "database", db };
	if (part === undefined) return { kind: "document", db, doc };
	if (part === "chunks" && chunk !== undefined && !rest.length)
		return { kind: "chunk", db, doc, chunk };
	return { kind: "unknown" };
}

const at = (path: string, query?: URLSearchParams) => {
	const text = query?.toString();
	return text ? `${path}?${text}` : path;
};

const segment = encodeURIComponent;

export const databaseHref = (db: string, query?: URLSearchParams) =>
	at(`/inspect/${segment(db)}`, query);

export const documentHref = (
	db: string,
	doc: string,
	query?: URLSearchParams,
) => at(`/inspect/${segment(db)}/${segment(doc)}`, query);

export const chunkHref = (
	db: string,
	doc: string,
	chunk: string,
	query?: URLSearchParams,
) =>
	at(`/inspect/${segment(db)}/${segment(doc)}/chunks/${segment(chunk)}`, query);

export const doctorHref = (db: string) => `/doctor/${segment(db)}`;

// The document list's state rides on every address of its database.
export interface ListState {
	q: string;
	filter: string;
	offset: number;
}

export function listState(query: URLSearchParams): ListState {
	return {
		q: query.get("list_q") ?? "",
		filter: query.get("list_filter") ?? "",
		offset: Number(query.get("list_offset")) || 0,
	};
}

export function listOf(query: URLSearchParams): URLSearchParams {
	const kept = new URLSearchParams();
	for (const key of ["list_q", "list_filter", "list_offset"]) {
		const value = query.get(key);
		if (value) kept.set(key, value);
	}
	return kept;
}

export function withList(
	query: URLSearchParams,
	list: ListState,
): URLSearchParams {
	const next = new URLSearchParams(query);
	const values = {
		list_q: list.q,
		list_filter: list.filter,
		list_offset: list.offset ? String(list.offset) : "",
	};
	for (const [key, value] of Object.entries(values)) {
		if (value) next.set(key, value);
		else next.delete(key);
	}
	return next;
}

export interface SearchInputs {
	q: string;
	dbs: string[];
	type: string;
	limit: string;
	filter: string;
}

export function searchOf(query: URLSearchParams): SearchInputs {
	return {
		q: query.get("q") ?? "",
		dbs: query.getAll("db"),
		type: query.get("type") ?? "hybrid",
		limit: query.get("limit") ?? "10",
		filter: query.get("filter") ?? "",
	};
}

export function searchHref(inputs: SearchInputs): string {
	const query = new URLSearchParams();
	if (inputs.q) query.set("q", inputs.q);
	for (const db of inputs.dbs) query.append("db", db);
	query.set("type", inputs.type);
	query.set("limit", inputs.limit);
	if (inputs.filter) query.set("filter", inputs.filter);
	return at("/search", query);
}
