/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Breadcrumb from "./Breadcrumb";

beforeEach(() => {
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) =>
			url.includes("/chunks/")
				? Response.json({ id: "c7", order: 7 })
				: Response.json({ id: "d1", title: "A report", uri: "doc://r" }),
		),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const list = new URLSearchParams("list_q=tea");

describe("Breadcrumb", () => {
	it("names the database alone on its page", () => {
		render(
			<Breadcrumb
				view={{ kind: "database", db: "orb" }}
				list={list}
				documentQuery={list}
			/>,
		);
		expect(screen.getByText("orb")).toBeTruthy();
		expect(screen.queryByRole("link")).toBeNull();
	});

	it("links the database from a document, and names the document", async () => {
		render(
			<Breadcrumb
				view={{ kind: "document", db: "orb", doc: "d1" }}
				list={list}
				documentQuery={list}
			/>,
		);
		await screen.findByText("A report");
		expect(screen.getByRole("link", { name: "orb" }).getAttribute("href")).toBe(
			"/inspect/orb?list_q=tea",
		);
		expect(screen.getAllByRole("link")).toHaveLength(1);
	});

	it("links the database and the document's view from a chunk", async () => {
		render(
			<Breadcrumb
				view={{ kind: "chunk", db: "orb", doc: "d1", chunk: "c7" }}
				list={list}
				documentQuery={new URLSearchParams("list_q=tea&tab=chunks")}
			/>,
		);
		await screen.findByText("Chunk 7");
		expect(
			screen.getByRole("link", { name: "A report" }).getAttribute("href"),
		).toBe("/inspect/orb/d1?list_q=tea&tab=chunks");
		expect(screen.getByRole("link", { name: "orb" }).getAttribute("href")).toBe(
			"/inspect/orb?list_q=tea",
		);
	});
});
