"use client";

import { Pencil, Plus, Trash2, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ingester } from "@/lib/ingestion";
import { type EditorRow, editorMetadata, editorRows } from "@/lib/inspect";

export interface EditableDocument {
	id: string;
	title: string | null;
	uri: string | null;
	metadata: Record<string, unknown>;
	created_at: string;
	updated_at: string;
}

const message = (error: unknown) =>
	error instanceof Error ? error.message : String(error);

// Why the ingester refuses to delete the document, read from its metadata. The
// ingester checks again: ownership can change after the page reads it.
function refusal(metadata: Record<string, unknown>): string | null {
	if (typeof metadata.source_id === "string")
		return `Source ${metadata.source_id} owns this document: its next sweep would ingest it again. Remove the file from the source instead.`;
	if (typeof metadata.parent_uri === "string")
		return `This is an attachment of ${metadata.parent_uri}. Delete that document, or remove the file from it and ingest it again.`;
	return null;
}

// Edits a document's title and editor metadata (`metadata.user`), and deletes
// it, through the ingester, the one writer of the database.
export default function DocumentEditor({
	source,
	document,
	onSaved,
	onDeleted,
}: {
	source: string;
	document: EditableDocument;
	onSaved: (document: EditableDocument) => void;
	onDeleted: () => void;
}) {
	const path = `databases/${encodeURIComponent(source)}/documents/${encodeURIComponent(document.id)}`;
	const [editing, setEditing] = useState(false);
	const [deleting, setDeleting] = useState(false);
	const [title, setTitle] = useState("");
	const [rows, setRows] = useState<EditorRow[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [busy, setBusy] = useState(false);
	const stored = document.metadata.user;
	const refused = refusal(document.metadata);
	const invalid =
		stored !== undefined &&
		(stored === null || typeof stored !== "object" || Array.isArray(stored));

	const openEditor = () => {
		setTitle(document.title ?? "");
		setRows(editorRows(stored));
		setError(null);
		setEditing(true);
	};

	const setRow = (index: number, change: Partial<EditorRow>) =>
		setRows((all) =>
			all.map((row, i) => (i === index ? { ...row, ...change } : row)),
		);

	const save = async () => {
		const user = editorMetadata(rows);
		if (typeof user === "string") {
			setError(user);
			return;
		}
		const edit: { title?: string; user: Record<string, unknown> } = { user };
		const trimmed = title.trim();
		if (!trimmed && document.title) {
			setError("A title cannot be empty.");
			return;
		}
		if (trimmed && trimmed !== document.title) edit.title = trimmed;
		setBusy(true);
		try {
			const saved = await ingester(path, {
				method: "PATCH",
				headers: { "content-type": "application/json" },
				body: JSON.stringify(edit),
			});
			setEditing(false);
			onSaved(saved as EditableDocument);
		} catch (err) {
			setError(message(err));
		} finally {
			setBusy(false);
		}
	};

	const remove = async () => {
		setBusy(true);
		try {
			await ingester(path, { method: "DELETE" });
			setDeleting(false);
			onDeleted();
		} catch (err) {
			setError(message(err));
		} finally {
			setBusy(false);
		}
	};

	return (
		<>
			<Button size="sm" variant="outline" onClick={openEditor}>
				<Pencil />
				Edit
			</Button>
			<span title={refused ?? undefined}>
				<Button
					size="sm"
					variant="outline"
					disabled={refused !== null}
					onClick={() => {
						setError(null);
						setDeleting(true);
					}}
				>
					<Trash2 />
					Delete
				</Button>
			</span>
			<Dialog open={editing} onOpenChange={setEditing}>
				<DialogContent>
					<DialogHeader>
						<DialogTitle>Edit document</DialogTitle>
						<DialogDescription>
							The title, and your own metadata, kept under{" "}
							<code>metadata.user</code> through re-ingests. A value marked JSON
							is stored as JSON, any other as text.
						</DialogDescription>
					</DialogHeader>
					<form
						className="space-y-3"
						onSubmit={(event) => {
							event.preventDefault();
							void save();
						}}
					>
						<div className="space-y-1 text-sm">
							<label htmlFor="document-title">Title</label>
							<Input
								id="document-title"
								value={title}
								onChange={(event) => setTitle(event.target.value)}
							/>
						</div>
						{invalid && (
							<p className="text-xs text-muted-foreground">
								The stored <code>metadata.user</code> is not an object. Saving
								replaces it.
							</p>
						)}
						<div className="space-y-1">
							{rows.map((row, index) => (
								// biome-ignore lint/suspicious/noArrayIndexKey: rows have no identity of their own
								<div key={index} className="flex gap-1">
									<Input
										aria-label="Key"
										placeholder="key"
										value={row.key}
										onChange={(event) =>
											setRow(index, { key: event.target.value })
										}
										className="w-1/3"
									/>
									<Input
										aria-label="Value"
										placeholder={row.json ? "JSON" : "value"}
										value={row.value}
										onChange={(event) =>
											setRow(index, { value: event.target.value })
										}
										className={row.json ? "font-mono" : undefined}
									/>
									<Button
										type="button"
										size="sm"
										variant={row.json ? "secondary" : "ghost"}
										aria-pressed={row.json}
										title="Store the value as JSON: a number, true, a list or an object"
										onClick={() => setRow(index, { json: !row.json })}
									>
										JSON
									</Button>
									<Button
										type="button"
										size="icon"
										variant="ghost"
										aria-label="Remove"
										title="Remove"
										onClick={() =>
											setRows((all) => all.filter((_, i) => i !== index))
										}
									>
										<X />
									</Button>
								</div>
							))}
							<Button
								type="button"
								size="sm"
								variant="ghost"
								onClick={() =>
									setRows((all) => [
										...all,
										{ key: "", value: "", json: false },
									])
								}
							>
								<Plus />
								Add metadata key-value
							</Button>
						</div>
						{error && <p className="text-sm text-destructive">{error}</p>}
						<DialogFooter>
							<Button type="submit" disabled={busy}>
								Save
							</Button>
						</DialogFooter>
					</form>
				</DialogContent>
			</Dialog>
			<Dialog open={deleting} onOpenChange={setDeleting}>
				<DialogContent>
					<DialogHeader>
						<DialogTitle>Delete document?</DialogTitle>
						<DialogDescription>
							{document.title ?? document.uri ?? document.id} and its
							attachments are removed from {source}.
						</DialogDescription>
					</DialogHeader>
					{error && <p className="text-sm text-destructive">{error}</p>}
					<DialogFooter>
						<Button
							variant="outline"
							onClick={() => setDeleting(false)}
							disabled={busy}
						>
							Cancel
						</Button>
						<Button variant="destructive" onClick={remove} disabled={busy}>
							Delete document
						</Button>
					</DialogFooter>
				</DialogContent>
			</Dialog>
		</>
	);
}
