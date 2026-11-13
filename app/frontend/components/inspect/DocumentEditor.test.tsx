/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DocumentEditor, { type EditableDocument } from "./DocumentEditor";

const stored: EditableDocument = {
	id: "d1",
	title: "Old title",
	uri: "doc://d1",
	metadata: { team: "x", user: { reviewer: "ana", tags: ["a"] } },
	created_at: "2026-11-13T10:00:00Z",
	updated_at: "2026-11-13T10:00:00Z",
};

let calls: { url: string; init: RequestInit }[];

function answer(status: number, body: unknown) {
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init: RequestInit) => {
			calls.push({ url, init });
			return Response.json(body, { status });
		}),
	);
}

beforeEach(() => {
	calls = [];
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("DocumentEditor", () => {
	it("saves the title and the editor metadata through the ingester", async () => {
		const edited = { ...stored, title: "New title", updated_at: "later" };
		answer(200, edited);
		const onSaved = vi.fn();
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="team/docs"
				document={stored}
				onSaved={onSaved}
				onDeleted={() => {}}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Edit" }));
		const title = screen.getByLabelText("Title");
		await user.clear(title);
		await user.type(title, "New title");
		await user.click(
			screen.getByRole("button", { name: "Add metadata key-value" }),
		);
		const keys = screen.getAllByLabelText("Key");
		const values = screen.getAllByLabelText("Value");
		await user.type(keys[2], "status");
		await user.type(values[2], "final");
		await user.click(screen.getByRole("button", { name: "Save" }));

		await waitFor(() => expect(onSaved).toHaveBeenCalledWith(edited));
		expect(calls[0].url).toBe(
			"/api/ingester/databases/team%2Fdocs/documents/d1",
		);
		expect(calls[0].init.method).toBe("PATCH");
		expect(JSON.parse(String(calls[0].init.body))).toEqual({
			title: "New title",
			user: { reviewer: "ana", tags: ["a"], status: "final" },
		});
		expect(screen.queryByLabelText("Title")).toBeNull();
	});

	it("removes a value and leaves an untouched title out of the edit", async () => {
		answer(200, stored);
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={stored}
				onSaved={() => {}}
				onDeleted={() => {}}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Edit" }));
		await user.click(screen.getAllByRole("button", { name: "Remove" })[0]);
		await user.click(screen.getByRole("button", { name: "Save" }));

		await waitFor(() => expect(calls).toHaveLength(1));
		expect(JSON.parse(String(calls[0].init.body))).toEqual({
			user: { tags: ["a"] },
		});
	});

	it("names invalid rows and sends nothing", async () => {
		answer(200, stored);
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={stored}
				onSaved={() => {}}
				onDeleted={() => {}}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Edit" }));
		const tags = screen.getAllByLabelText("Value")[1];
		await user.clear(tags);
		await user.type(tags, "[[not json");
		await user.click(screen.getByRole("button", { name: "Save" }));

		expect(await screen.findByText("tags is not valid JSON.")).toBeTruthy();
		expect(calls).toHaveLength(0);
	});

	it("stores a value marked as JSON as JSON", async () => {
		answer(200, stored);
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={{ ...stored, metadata: {} }}
				onSaved={() => {}}
				onDeleted={() => {}}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Edit" }));
		await user.click(
			screen.getByRole("button", { name: "Add metadata key-value" }),
		);
		await user.click(
			screen.getByRole("button", { name: "Add metadata key-value" }),
		);
		const keys = screen.getAllByLabelText("Key");
		const values = screen.getAllByLabelText("Value");
		await user.type(keys[0], "pages");
		await user.type(values[0], "42");
		await user.click(screen.getAllByRole("button", { name: "JSON" })[0]);
		await user.type(keys[1], "count");
		await user.type(values[1], "42");
		await user.click(screen.getByRole("button", { name: "Save" }));

		await waitFor(() => expect(calls).toHaveLength(1));
		expect(JSON.parse(String(calls[0].init.body)).user).toEqual({
			pages: 42,
			count: "42",
		});
	});

	it("refuses to clear a title", async () => {
		answer(200, stored);
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={stored}
				onSaved={() => {}}
				onDeleted={() => {}}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Edit" }));
		await user.clear(screen.getByLabelText("Title"));
		await user.click(screen.getByRole("button", { name: "Save" }));

		expect(await screen.findByText("A title cannot be empty.")).toBeTruthy();
		expect(calls).toHaveLength(0);
	});

	it("says a stored value that is not an object is replaced", async () => {
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={{ ...stored, metadata: { user: "legacy" } }}
				onSaved={() => {}}
				onDeleted={() => {}}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Edit" }));

		expect(screen.getByText(/is not an object/)).toBeTruthy();
		expect(screen.queryAllByLabelText("Key")).toHaveLength(0);
	});

	it("shows the ingester's refusal and keeps the form open", async () => {
		answer(404, { detail: "database not ingested" });
		const onSaved = vi.fn();
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={stored}
				onSaved={onSaved}
				onDeleted={() => {}}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Edit" }));
		await user.click(screen.getByRole("button", { name: "Save" }));

		expect(await screen.findByText("database not ingested")).toBeTruthy();
		expect(onSaved).not.toHaveBeenCalled();
		expect(screen.getByLabelText("Title")).toBeTruthy();
	});

	it("deletes the document once confirmed", async () => {
		answer(200, { document_id: "d1", deleted: true });
		const onDeleted = vi.fn();
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={stored}
				onSaved={() => {}}
				onDeleted={onDeleted}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Delete" }));
		expect(calls).toHaveLength(0);
		await user.click(screen.getByRole("button", { name: "Delete document" }));

		await waitFor(() => expect(onDeleted).toHaveBeenCalled());
		expect(calls[0].url).toBe("/api/ingester/databases/docs/documents/d1");
		expect(calls[0].init.method).toBe("DELETE");
	});

	it.each([
		[{ source_id: "mitos" }, "Source mitos owns this document"],
		[{ parent_uri: "file:///a.pdf" }, "an attachment of file:///a.pdf"],
	])(
		"disables delete on a source-owned document or an attachment",
		async (metadata, reason) => {
			render(
				<DocumentEditor
					source="docs"
					document={{ ...stored, metadata }}
					onSaved={() => {}}
					onDeleted={() => {}}
				/>,
			);

			const remove = screen.getByRole("button", { name: "Delete" });
			expect((remove as HTMLButtonElement).disabled).toBe(true);
			expect(remove.parentElement?.title).toContain(reason);
		},
	);

	it("shows why the ingester refused a delete", async () => {
		answer(409, {
			detail:
				"Document d1 belongs to source fs:docs. Remove it from the source instead.",
		});
		const onDeleted = vi.fn();
		const user = userEvent.setup();
		render(
			<DocumentEditor
				source="docs"
				document={stored}
				onSaved={() => {}}
				onDeleted={onDeleted}
			/>,
		);

		await user.click(screen.getByRole("button", { name: "Delete" }));
		await user.click(screen.getByRole("button", { name: "Delete document" }));

		expect(await screen.findByText(/belongs to source fs:docs/)).toBeTruthy();
		expect(onDeleted).not.toHaveBeenCalled();
	});
});
