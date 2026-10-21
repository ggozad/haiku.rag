"use client";

import { useEffect, useRef, useState } from "react";
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
import { load } from "@/lib/useJson";

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

interface Listing {
	documents?: ListedDocument[];
	totals?: Record<string, number>;
	errors?: Record<string, string>;
}

interface Further {
	pages: ListedDocument[][];
	total?: number;
	loading?: boolean;
	error?: string;
}

const PAGE = 50;
// Typing waits this long before the server is asked.
const SETTLE_MS = 250;

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
	const [query, setQuery] = useState("");
	// The first page of every database, then each database's further pages.
	const [first, setFirst] = useState<Listing | null>(null);
	const [failure, setFailure] = useState<string | null>(null);
	const [further, setFurther] = useState<Record<string, Further>>({});
	const requests = useRef(new Set<AbortController>());
	const [chosen, setChosen] = useState<Set<string>>(new Set());

	useEffect(() => {
		const timer = setTimeout(() => setQuery(term.trim()), SETTLE_MS);
		return () => clearTimeout(timer);
	}, [term]);

	const request = (
		params: Record<string, string>,
		on: Parameters<typeof load<Listing>>[2],
	) => {
		const search = new URLSearchParams({ ...params, limit: String(PAGE) });
		if (query) search.set("q", query);
		const controller = new AbortController();
		requests.current.add(controller);
		void load<Listing>(
			`/api/backend/api/documents?${search}`,
			controller.signal,
			{
				...on,
				done: () => {
					requests.current.delete(controller);
					on.done();
				},
			},
		);
	};

	// Listed on every open, so new and deleted documents show, and found by
	// title or URI on the server.
	// biome-ignore lint/correctness/useExhaustiveDependencies: request reads query
	useEffect(() => {
		if (!open) return;
		setFirst(null);
		setFailure(null);
		setFurther({});
		request(
			{ offset: "0" },
			{ data: setFirst, error: setFailure, done: () => {} },
		);
		const pending = requests.current;
		return () => {
			for (const controller of pending) controller.abort();
			pending.clear();
		};
	}, [open, query]);

	// One database's next page. A failed page is asked for again next time.
	const showMore = (source: string) => {
		const pages = further[source]?.pages ?? [];
		const update = (change: Partial<Further>) =>
			setFurther((kept) => ({
				...kept,
				[source]: { ...(kept[source] ?? { pages: [] }), ...change },
			}));
		update({ loading: true, error: undefined });
		request(
			{ source, offset: String(PAGE * (pages.length + 1)) },
			{
				data: (page) =>
					update({
						pages: [...pages, page.documents ?? []],
						total: page.totals?.[source],
					}),
				error: (message) => update({ error: message }),
				done: () => update({ loading: false }),
			},
		);
	};

	useEffect(() => {
		if (!open) return;
		setChosen(new Set(selection.map(keyOf)));
		setTerm("");
		setQuery("");
	}, [open, selection]);

	const sources = [
		...new Set((first?.documents ?? []).map((doc) => doc.source)),
	].map((source) => {
		const opening = (first?.documents ?? []).filter(
			(doc) => doc.source === source,
		);
		const pages = further[source]?.pages ?? [];
		// Pages are read at offsets, so a write between two of them can list a
		// document twice.
		const docs = [
			...new Map(
				[opening, ...pages].flat().map((doc) => [keyOf(doc), doc]),
			).values(),
		].sort((a, b) => nameOf(a).localeCompare(nameOf(b)));
		const total =
			further[source]?.total ?? first?.totals?.[source] ?? docs.length;
		// More while the last page came back full: the total can count
		// documents shifted by a write past the pages already read.
		const more =
			total > docs.length && (pages.at(-1) ?? opening).length === PAGE;
		return { source, docs, total, more, ...further[source] };
	});
	const errors = failure ? { backend: failure } : (first?.errors ?? {});
	const loading = open && first === null && failure === null;

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
					{!loading && sources.length === 0 && (
						<p className="text-sm text-muted-foreground">No documents found.</p>
					)}
					{sources.map(({ source, docs, total, more, loading, error }) => (
						<section key={source} className="space-y-1">
							<h3 className="flex justify-between text-xs font-medium uppercase tracking-wide text-muted-foreground">
								<span>{source}</span>
								<span className="normal-case tabular-nums">
									{docs.length} of {total}
								</span>
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
							{error && <p className="text-xs text-destructive">{error}</p>}
							{more && (
								<Button
									variant="ghost"
									size="sm"
									disabled={loading}
									onClick={() => showMore(source)}
								>
									Show more
								</Button>
							)}
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
