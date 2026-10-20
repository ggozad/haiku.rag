"use client";

import {
	useParams,
	usePathname,
	useRouter,
	useSearchParams,
} from "next/navigation";
import { useEffect, useState } from "react";
import { Estimate, useEstimate } from "@/components/doctor/MapJob";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import {
	chunkHref,
	databaseHref,
	documentHref,
	type InspectView,
	inspectView,
	listOf,
	listState,
	searchHref,
	searchOf,
	withList,
} from "@/lib/addresses";
import { useDatabases } from "@/lib/databases";
import { formatRelativeTime } from "@/lib/format";
import { useJson } from "@/lib/useJson";
import { cn } from "@/lib/utils";
import Breadcrumb from "./Breadcrumb";
import ChunkView, { type ChunkRef } from "./ChunkView";
import CorpusMap from "./CorpusMap";
import DocumentView, { type DocumentState, OPENING } from "./DocumentView";
import InfoDialog from "./InfoDialog";
import SearchPanel from "./SearchPanel";
import ShowOnMap from "./ShowOnMap";

interface Listing {
	total: number;
	documents: {
		id: string;
		title: string | null;
		uri: string | null;
		source_id: string | null;
		content_type: string | null;
		created_at: string;
		updated_at: string;
	}[];
}

const PAGE = 50;

// A database's map, with `doc` selected, keeping the document list's query.
function mapHref(db: string, list: URLSearchParams, doc?: string) {
	const query = new URLSearchParams(list);
	query.set("view", "map");
	if (doc) query.set("doc", doc);
	return databaseHref(db, query);
}

function MapView({
	source,
	selectedId,
	onSelect,
	documentHref,
}: {
	source: string;
	selectedId: string | null;
	onSelect: (id: string | null) => void;
	documentHref: (id: string) => string;
}) {
	const estimate = useEstimate(source);
	return (
		<div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
			<Estimate text={estimate.text} />
			<CorpusMap
				key={source}
				source={source}
				ready={estimate.ready}
				selectedId={selectedId}
				onSelect={onSelect}
				documentHref={documentHref}
			/>
		</div>
	);
}

function documentState(query: URLSearchParams): DocumentState {
	const number = (key: string) =>
		query.has(key) ? Number(query.get(key)) : null;
	return {
		tab: query.get("tab") ?? OPENING.tab,
		offset: number("offset") ?? OPENING.offset,
		page: number("page"),
		item: number("item"),
	};
}

function withDocument(
	query: URLSearchParams,
	change: Partial<DocumentState>,
): URLSearchParams {
	const next = new URLSearchParams(query);
	for (const [key, value] of Object.entries(change)) {
		const opening = OPENING[key as keyof DocumentState];
		if (value === null || value === opening) next.delete(key);
		else next.set(key, String(value));
	}
	return next;
}

// Every view, and its state, is read from the address.
export default function InspectPage() {
	const { info } = useDatabases();
	const databases = info?.databases ?? [];
	const available = databases.filter((db) => db.available);
	const router = useRouter();
	const pathname = usePathname();
	const query = useSearchParams();
	const path = useParams<{ path?: string[] }>().path;
	const view: InspectView =
		pathname === "/search" ? { kind: "search" } : inspectView(path);
	const [infoOpen, setInfoOpen] = useState(false);

	// The search page keeps showing the list of the database last open, and the
	// database pages link back to the search last run.
	const [lastList, setLastList] = useState<{
		db: string;
		query: string;
	} | null>(null);
	const [lastSearch, setLastSearch] = useState("/search");
	const here = query.toString();
	const viewDb = "db" in view ? view.db : null;
	const listHere = listOf(query).toString();
	useEffect(() => {
		if (viewDb !== null) setLastList({ db: viewDb, query: listHere });
	}, [viewDb, listHere]);
	const searching = view.kind === "search";
	useEffect(() => {
		if (searching) setLastSearch(here ? `/search?${here}` : "/search");
	}, [searching, here]);

	const first = available[0] ?? databases[0];
	useEffect(() => {
		if (view.kind === "home" && first) router.replace(databaseHref(first.name));
	}, [view.kind, first, router]);

	const source = "db" in view ? view.db : (lastList?.db ?? first?.name ?? null);
	const listQuery =
		"db" in view
			? query
			: new URLSearchParams(lastList?.db === source ? lastList.query : "");
	const list = listState(listQuery);
	const [text, setText] = useState(list.q);
	const [filterText, setFilterText] = useState(list.filter);
	useEffect(() => {
		setText(list.q);
		setFilterText(list.filter);
	}, [list.q, list.filter]);

	const params = new URLSearchParams({
		limit: String(PAGE),
		offset: String(list.offset),
	});
	if (list.q) params.set("q", list.q);
	if (list.filter) params.set("filter", list.filter);
	// A database the app cannot open offers Info only.
	const selected = databases.find((db) => db.name === source);
	const listing = useJson<Listing>(
		source && selected?.available
			? `/api/backend/api/inspect/${encodeURIComponent(source)}/documents?${params}`
			: null,
	);

	const showList = (next: typeof list) => {
		if (source === null) return;
		router.push(
			"db" in view
				? `${pathname}?${withList(query, next)}`
				: databaseHref(source, withList(listQuery, next)),
		);
	};

	// The query an address of `db` carries from here.
	const carried = (db: string) =>
		db === source ? listOf(listQuery) : undefined;

	const openChunk = (ref: ChunkRef) => {
		const keep =
			"doc" in view && view.db === ref.source && view.doc === ref.documentId
				? new URLSearchParams(query)
				: carried(ref.source);
		keep?.delete("view");
		router.push(chunkHref(ref.source, ref.documentId, ref.chunkId, keep));
	};

	const mapOpen = view.kind === "database" && query.get("view") === "map";

	const documentQuery = new URLSearchParams(query);
	documentQuery.delete("view");

	return (
		<div className="grid h-full grid-cols-1 md:grid-cols-[22rem_minmax(0,1fr)]">
			<aside className="flex min-h-0 flex-col border-r">
				<div className="space-y-2 border-b p-3">
					<div className="flex gap-1">
						<Button
							size="sm"
							variant={
								view.kind === "search" || mapOpen ? "ghost" : "secondary"
							}
							disabled={!source}
							onClick={() =>
								source && router.push(databaseHref(source, listOf(listQuery)))
							}
						>
							Documents
						</Button>
						<Button
							size="sm"
							variant={mapOpen ? "secondary" : "ghost"}
							disabled={!selected}
							onClick={() =>
								source && router.push(mapHref(source, listOf(listQuery)))
							}
						>
							Map
						</Button>
						<Button
							size="sm"
							variant={view.kind === "search" ? "secondary" : "ghost"}
							onClick={() => router.push(lastSearch)}
						>
							Search
						</Button>
						<Button
							size="sm"
							variant="ghost"
							className="ml-auto"
							disabled={!source}
							onClick={() => setInfoOpen(true)}
						>
							Info
						</Button>
					</div>
					<Select
						value={source ?? ""}
						onValueChange={(name) => router.push(databaseHref(name))}
					>
						<SelectTrigger className="w-full" aria-label="Database">
							<SelectValue placeholder="Database" />
						</SelectTrigger>
						<SelectContent>
							{(info?.databases ?? []).map((db) => (
								<SelectItem key={db.name} value={db.name}>
									{db.available
										? `${db.name} · ${db.documents} documents`
										: `${db.name} · unavailable, Info only`}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
					{selected && !selected.available && (
						<p className="text-xs text-muted-foreground">
							{selected.name} cannot be opened
							{selected.error ? `: ${selected.error}` : ""}. Info still reads
							it.
						</p>
					)}
					<form
						hidden={!selected?.available}
						onSubmit={(event) => {
							event.preventDefault();
							showList({
								q: text.trim(),
								filter: filterText.trim(),
								offset: 0,
							});
						}}
						className="space-y-2"
					>
						<Input
							id="inspect-search"
							placeholder="Title or URI contains"
							value={text}
							onChange={(event) => setText(event.target.value)}
						/>
						<Input
							id="inspect-filter"
							placeholder="SQL filter, e.g. metadata LIKE '%report%'"
							value={filterText}
							onChange={(event) => setFilterText(event.target.value)}
							className="font-mono text-xs"
						/>
						<Button type="submit" size="sm" className="w-full">
							Apply
						</Button>
					</form>
					{listing.error && (
						<p className="text-xs text-destructive">{listing.error}</p>
					)}
				</div>
				<ul className="min-h-0 flex-1 overflow-y-auto">
					{listing.data?.documents.map((doc) => (
						<li key={doc.id}>
							<button
								type="button"
								onClick={() =>
									source &&
									router.push(documentHref(source, doc.id, listOf(listQuery)))
								}
								className={cn(
									"w-full border-b px-3 py-2 text-left text-sm hover:bg-muted",
									"doc" in view && doc.id === view.doc && "bg-muted",
								)}
							>
								<div className="truncate font-medium">
									{doc.title || doc.uri || doc.id}
								</div>
								{doc.title && doc.uri && (
									<div className="truncate text-xs text-muted-foreground">
										{doc.uri}
									</div>
								)}
								<div className="flex gap-2 truncate text-xs text-muted-foreground">
									{doc.source_id && <span>{doc.source_id}</span>}
									{doc.content_type && <span>{doc.content_type}</span>}
									<span>
										created {formatRelativeTime(doc.created_at, true)}, updated{" "}
										{formatRelativeTime(doc.updated_at, true)}
									</span>
								</div>
							</button>
						</li>
					))}
				</ul>
				{listing.data && (
					<div className="flex items-center gap-2 border-t p-2 text-xs">
						<Button
							size="sm"
							variant="outline"
							disabled={list.offset === 0}
							onClick={() =>
								showList({ ...list, offset: Math.max(0, list.offset - PAGE) })
							}
						>
							Previous
						</Button>
						<span className="flex-1 text-center text-muted-foreground tabular-nums">
							{listing.data.total
								? `${list.offset + 1}–${list.offset + listing.data.documents.length} of ${listing.data.total}`
								: "No documents"}
						</span>
						<Button
							size="sm"
							variant="outline"
							disabled={list.offset + PAGE >= listing.data.total}
							onClick={() => showList({ ...list, offset: list.offset + PAGE })}
						>
							Next
						</Button>
					</div>
				)}
			</aside>
			<section className="flex min-h-0 min-w-0 flex-col">
				{"db" in view && selected && (
					<Breadcrumb
						view={view}
						list={listOf(listQuery)}
						documentQuery={documentQuery}
					/>
				)}
				{view.kind === "search" ? (
					<SearchPanel
						key={here}
						databases={available}
						inputs={searchOf(query)}
						onSearch={(inputs) => router.push(searchHref(inputs))}
						onOpen={openChunk}
					/>
				) : "db" in view && info && !selected ? (
					<p className="p-6 text-sm text-muted-foreground">
						No database is named {view.db}.
					</p>
				) : mapOpen && selected ? (
					<MapView
						source={view.db}
						selectedId={query.get("doc")}
						onSelect={(id) => {
							const next = new URLSearchParams(query);
							if (id) next.set("doc", id);
							else next.delete("doc");
							router.replace(databaseHref(view.db, next));
						}}
						documentHref={(id) => documentHref(view.db, id, listOf(listQuery))}
					/>
				) : view.kind === "chunk" ? (
					<ChunkView
						key={`${view.db}:${view.chunk}`}
						chunk={{
							source: view.db,
							documentId: view.doc,
							chunkId: view.chunk,
						}}
						view={query.get("view") ?? "details"}
						onView={(next) => {
							const kept = new URLSearchParams(query);
							if (next === "details") kept.delete("view");
							else kept.set("view", next);
							router.push(chunkHref(view.db, view.doc, view.chunk, kept));
						}}
						onOpen={openChunk}
						chunkHref={(id) => chunkHref(view.db, view.doc, id, query)}
						documentHref={(item) =>
							documentHref(
								view.db,
								view.doc,
								withDocument(documentQuery, { tab: "content", item }),
							)
						}
					/>
				) : "doc" in view ? (
					<DocumentView
						key={`${view.db}:${view.doc}`}
						source={view.db}
						documentId={view.doc}
						state={documentState(documentQuery)}
						onState={(change) =>
							router.push(
								documentHref(
									view.db,
									view.doc,
									withDocument(documentQuery, change),
								),
							)
						}
						actions={
							<ShowOnMap
								source={view.db}
								documentId={view.doc}
								href={mapHref(view.db, listOf(listQuery), view.doc)}
							/>
						}
						onOpenChunk={(chunkId) =>
							openChunk({
								source: view.db,
								documentId: view.doc,
								chunkId,
							})
						}
					/>
				) : view.kind === "unknown" ? (
					<p className="p-6 text-sm text-muted-foreground">
						The inspector has no page at this address.
					</p>
				) : (
					<p className="p-6 text-sm text-muted-foreground">
						Pick a document to see its content, outline and chunks.
					</p>
				)}
			</section>
			{source && (
				<InfoDialog
					source={source}
					open={infoOpen}
					onOpenChange={setInfoOpen}
				/>
			)}
		</div>
	);
}
