"use client";

import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { databaseHref, documentHref } from "@/lib/addresses";
import { useJson } from "@/lib/useJson";

type Place =
	| { kind: "database"; db: string }
	| { kind: "document"; db: string; doc: string }
	| { kind: "chunk"; db: string; doc: string; chunk: string };

const base = (db: string) =>
	`/api/backend/api/inspect/${encodeURIComponent(db)}`;

// The database, document and chunk open, each above the current one a link.
// `list` is the document list's query, `documentQuery` the document's view.
export default function Breadcrumb({
	view,
	list,
	documentQuery,
}: {
	view: Place;
	list: URLSearchParams;
	documentQuery: URLSearchParams;
}) {
	const doc = useJson<{ title: string | null; uri: string | null }>(
		"doc" in view
			? `${base(view.db)}/documents/${encodeURIComponent(view.doc)}`
			: null,
	);
	const chunk = useJson<{ order: number }>(
		view.kind === "chunk"
			? `${base(view.db)}/chunks/${encodeURIComponent(view.chunk)}`
			: null,
	);
	const crumbs: { label: string; href?: string }[] = [
		{
			label: view.db,
			href: view.kind === "database" ? undefined : databaseHref(view.db, list),
		},
	];
	if ("doc" in view)
		crumbs.push({
			label: doc.data?.title || doc.data?.uri || view.doc,
			href:
				view.kind === "chunk"
					? documentHref(view.db, view.doc, documentQuery)
					: undefined,
		});
	if (view.kind === "chunk")
		crumbs.push({
			label: chunk.data ? `Chunk ${chunk.data.order}` : "Chunk",
		});

	return (
		<nav aria-label="Breadcrumb" className="border-b px-4 py-2 text-sm">
			<ol className="flex min-w-0 items-center gap-1">
				{crumbs.map((crumb, n) => (
					<li
						key={`${n}:${crumb.label}`}
						className="flex min-w-0 items-center gap-1"
					>
						{n > 0 && (
							<ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
						)}
						{crumb.href ? (
							<Link
								href={crumb.href}
								className="truncate text-muted-foreground hover:text-foreground hover:underline"
							>
								{crumb.label}
							</Link>
						) : (
							<span className="truncate font-medium" aria-current="page">
								{crumb.label}
							</span>
						)}
					</li>
				))}
			</ol>
		</nav>
	);
}
