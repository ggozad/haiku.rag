import { describe, expect, it } from "vitest";
import {
	chunkHref,
	documentHref,
	inspectView,
	listOf,
	searchHref,
	searchOf,
	withList,
} from "./addresses";

describe("inspectView", () => {
	it("reads each inspector address", () => {
		expect(inspectView(undefined)).toEqual({ kind: "home" });
		expect(inspectView(["orb"])).toEqual({ kind: "database", db: "orb" });
		expect(inspectView(["orb", "d1"])).toEqual({
			kind: "document",
			db: "orb",
			doc: "d1",
		});
		expect(inspectView(["orb", "d1", "chunks", "c1"])).toEqual({
			kind: "chunk",
			db: "orb",
			doc: "d1",
			chunk: "c1",
		});
		expect(inspectView(["search"])).toEqual({ kind: "database", db: "search" });
	});

	it("reads an address it does not know as unknown", () => {
		expect(inspectView(["orb", "d1", "pages", "c1"])).toEqual({
			kind: "unknown",
		});
		expect(inspectView(["orb", "d1", "chunks"])).toEqual({ kind: "unknown" });
	});
});

describe("addresses", () => {
	it("encodes every segment", () => {
		expect(documentHref("my db", "a/b")).toBe("/inspect/my%20db/a%2Fb");
		expect(chunkHref("orb", "d1", "c 1")).toBe("/inspect/orb/d1/chunks/c%201");
	});

	it("carries a query when given one", () => {
		expect(
			chunkHref("orb", "d1", "c1", new URLSearchParams({ view: "agent" })),
		).toBe("/inspect/orb/d1/chunks/c1?view=agent");
	});

	it("keeps only the list's state from a query", () => {
		const query = new URLSearchParams(
			"tab=chunks&list_q=tea&list_filter=x%3D1&list_offset=50&offset=100",
		);
		expect(listOf(query).toString()).toBe(
			"list_q=tea&list_filter=x%3D1&list_offset=50",
		);
	});

	it("replaces the list's state and keeps the rest", () => {
		const query = new URLSearchParams("tab=chunks&list_q=tea&list_offset=50");
		expect(
			withList(query, { q: "milk", filter: "", offset: 0 }).toString(),
		).toBe("tab=chunks&list_q=milk");
	});
});

describe("search addresses", () => {
	it("round-trips every input, one db per database", () => {
		const inputs = {
			q: "tea",
			dbs: ["orb", "medic"],
			type: "fts",
			limit: "20",
			filter: "uri LIKE '%a%'",
		};
		const href = searchHref(inputs);
		expect(href).toBe(
			"/search?q=tea&db=orb&db=medic&type=fts&limit=20&filter=uri+LIKE+%27%25a%25%27",
		);
		expect(searchOf(new URL(href, "http://app").searchParams)).toEqual(inputs);
	});

	it("fills the defaults a bare address leaves out", () => {
		expect(searchOf(new URLSearchParams())).toEqual({
			q: "",
			dbs: [],
			type: "hybrid",
			limit: "10",
			filter: "",
		});
	});
});
