"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import type { DocumentRef } from "@/lib/chat";
import { useJson } from "@/lib/useJson";

interface ListedDocument {
	id: string;
	title: string | null;
	uri: string | null;
	source: string;
}

const keyOf = (doc: DocumentRef) => JSON.stringify([doc.source, doc.id]);
const refOf = (key: string): DocumentRef => {
	const [source, id] = JSON.parse(key);
	return { source, id };
};
const nameOf = (doc: ListedDocument) => doc.title || doc.uri || doc.id;

export default function FilterDialog({
	open,
	onOpenChange,
	selection,
	onApply,
}: {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	selection: DocumentRef[];
	onApply: (selection: DocumentRef[]) => void;
}) {
	const [term, setTerm] = useState("");
	const [chosen, setChosen] = useState<Set<string>>(new Set());

	// Listed on every open, so new and deleted documents show.
	const listing = useJson<{
		documents?: ListedDocument[];
		errors?: Record<string, string>;
	}>(open ? "/api/backend/api/documents" : null);
	const documents = listing.data?.documents ?? [];
	const errors = listing.error
		? { backend: listing.error }
		: (listing.data?.errors ?? {});
	const loading = listing.loading;

	useEffect(() => {
		if (!open) return;
		setChosen(new Set(selection.map(keyOf)));
		setTerm("");
	}, [open, selection]);

	const grouped = useMemo(() => {
		const needle = term.trim().toLowerCase();
		const groups = new Map<string, ListedDocument[]>();
		for (const doc of documents) {
			if (needle && !nameOf(doc).toLowerCase().includes(needle)) continue;
			const group = groups.get(doc.source) ?? [];
			group.push(doc);
			groups.set(doc.source, group);
		}
		for (const group of groups.values()) {
			group.sort((a, b) => nameOf(a).localeCompare(nameOf(b)));
		}
		return groups;
	}, [documents, term]);

	const toggle = (key: string) =>
		setChosen((prev) => {
			const next = new Set(prev);
			if (next.has(key)) next.delete(key);
			else next.add(key);
			return next;
		});

	// Built from the chosen keys, not from the listing, so a selected document
	// whose database could not be listed stays selected.
	const apply = () => {
		onApply([...chosen].map(refOf));
		onOpenChange(false);
	};

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="sm:max-w-lg">
				<DialogHeader>
					<DialogTitle>Search only selected documents</DialogTitle>
					<DialogDescription>
						With none selected, every document is searched.
					</DialogDescription>
				</DialogHeader>
				<Input
					id="filter-search"
					placeholder="Find a document"
					value={term}
					onChange={(event) => setTerm(event.target.value)}
				/>
				{Object.entries(errors).map(([name, reason]) => (
					<p key={name} className="text-xs text-destructive">
						{name}: {reason}
					</p>
				))}
				<div className="max-h-80 space-y-4 overflow-y-auto">
					{loading && (
						<p className="text-sm text-muted-foreground">Loading documents…</p>
					)}
					{!loading && grouped.size === 0 && (
						<p className="text-sm text-muted-foreground">No documents found.</p>
					)}
					{[...grouped.entries()].map(([source, docs]) => (
						<section key={source} className="space-y-1">
							<h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
								{source}
							</h3>
							{docs.map((doc) => {
								const key = keyOf(doc);
								return (
									<label
										key={key}
										htmlFor={`doc-${doc.source}-${doc.id}`}
										className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 text-sm hover:bg-muted"
									>
										<Checkbox
											id={`doc-${doc.source}-${doc.id}`}
											checked={chosen.has(key)}
											onCheckedChange={() => toggle(key)}
										/>
										<span className="truncate">{nameOf(doc)}</span>
									</label>
								);
							})}
						</section>
					))}
				</div>
				<DialogFooter className="items-center sm:justify-between">
					<span className="text-xs text-muted-foreground">
						{chosen.size ? `${chosen.size} selected` : "All documents"}
					</span>
					<div className="flex gap-2">
						<Button variant="outline" onClick={() => setChosen(new Set())}>
							Clear
						</Button>
						<Button onClick={apply} disabled={loading}>
							Apply
						</Button>
					</div>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}
