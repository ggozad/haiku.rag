// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import DatabaseTables, { type DatabaseInfo } from "./DatabaseInfo";

afterEach(cleanup);

const info: DatabaseInfo = {
	path: "/data/docs.lancedb",
	exists: true,
	stored_version: "0.107.0",
	embeddings: { provider: "openai", name: "e", vector_dim: 3 },
	tables: [
		{
			name: "chunks",
			exists: true,
			num_rows: 300,
			total_bytes: 2048,
			num_versions: 4,
			latest_version_at: null,
		},
		{
			name: "document_items",
			exists: false,
			num_rows: 0,
			total_bytes: 0,
			num_versions: 0,
			latest_version_at: null,
		},
	],
	vector_index: { exists: true, indexed_rows: 256, unindexed_rows: 44 },
	pending_migrations: [{ version: "0.111.0", description: "Rebuild indexes" }],
	packages: { haiku_rag: "0.110.0" },
};

describe("DatabaseTables", () => {
	it("shows tables, the vector index and pending migrations", () => {
		render(<DatabaseTables info={info} />);
		const chunks = screen.getByText("chunks").closest("tr");
		expect(chunks?.textContent).toBe("chunks3002.0 KB4");
		expect(screen.getByText("document_items").closest("tr")?.textContent).toBe(
			"document_itemsmissing",
		);
		expect(
			screen.getByText("Vector index: 256 rows indexed, 44 not yet"),
		).toBeTruthy();
		expect(screen.getByText("0.111.0: Rebuild indexes")).toBeTruthy();
	});

	it("says when the database does not exist", () => {
		render(<DatabaseTables info={{ ...info, exists: false }} />);
		expect(screen.getByText("The database does not exist.")).toBeTruthy();
		expect(screen.queryByRole("table")).toBeNull();
	});
});
