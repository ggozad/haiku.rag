"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { useDatabases } from "@/lib/databases";
import { formatRelativeTime } from "@/lib/format";
import { useJson } from "@/lib/useJson";
import { cn } from "@/lib/utils";
import ChunkDialog, { type ChunkRef } from "./ChunkDialog";
import DocumentView from "./DocumentView";
import InfoDialog from "./InfoDialog";
import SearchPanel from "./SearchPanel";

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

export default function InspectPage() {
	const { info } = useDatabases();
	const available = info?.databases.filter((db) => db.available) ?? [];
	const [source, setSource] = useState<string | null>(null);
	const [text, setText] = useState("");
	const [query, setQuery] = useState("");
	const [filterText, setFilterText] = useState("");
	const [filter, setFilter] = useState("");
	const [offset, setOffset] = useState(0);
	const [documentId, setDocumentId] = useState<string | null>(null);
	const [chunk, setChunk] = useState<ChunkRef | null>(null);
	const [mode, setMode] = useState<"documents" | "search">("documents");
	const [infoOpen, setInfoOpen] = useState(false);

	// The map links here with ?source=…&document=….
	const linked = useSearchParams();
	useEffect(() => {
		if (source !== null || !available.length) return;
		const wanted = linked.get("source");
		const known = available.find((db) => db.name === wanted);
		setSource(known ? known.name : available[0].name);
		if (known) setDocumentId(linked.get("document"));
	}, [source, available, linked]);

	const params = new URLSearchParams({
		limit: String(PAGE),
		offset: String(offset),
	});
	if (query) params.set("q", query);
	if (filter) params.set("filter", filter);
	// A database the app cannot open offers Info only.
	const selected = info?.databases.find((db) => db.name === source);
	const listing = useJson<Listing>(
		source && selected?.available
			? `/api/backend/api/inspect/${encodeURIComponent(source)}/documents?${params}`
			: null,
	);

	const choose = (name: string) => {
		setSource(name);
		setOffset(0);
		setDocumentId(null);
	};

	return (
		<div className="grid h-full grid-cols-1 md:grid-cols-[22rem_minmax(0,1fr)]">
			<aside className="flex min-h-0 flex-col border-r">
				<div className="space-y-2 border-b p-3">
					<div className="flex gap-1">
						<Button
							size="sm"
							variant={mode === "documents" ? "secondary" : "ghost"}
							onClick={() => setMode("documents")}
						>
							Documents
						</Button>
						<Button
							size="sm"
							variant={mode === "search" ? "secondary" : "ghost"}
							onClick={() => setMode("search")}
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
					<Select value={source ?? ""} onValueChange={choose}>
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
							setQuery(text.trim());
							setFilter(filterText.trim());
							setOffset(0);
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
								onClick={() => setDocumentId(doc.id)}
								className={cn(
									"w-full border-b px-3 py-2 text-left text-sm hover:bg-muted",
									doc.id === documentId && "bg-muted",
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
							disabled={offset === 0}
							onClick={() => setOffset(Math.max(0, offset - PAGE))}
						>
							Previous
						</Button>
						<span className="flex-1 text-center text-muted-foreground tabular-nums">
							{listing.data.total
								? `${offset + 1}–${offset + listing.data.documents.length} of ${listing.data.total}`
								: "No documents"}
						</span>
						<Button
							size="sm"
							variant="outline"
							disabled={offset + PAGE >= listing.data.total}
							onClick={() => setOffset(offset + PAGE)}
						>
							Next
						</Button>
					</div>
				)}
			</aside>
			<section className="min-h-0 min-w-0">
				{mode === "search" ? (
					<SearchPanel databases={available} onOpen={setChunk} />
				) : source && documentId ? (
					<DocumentView
						key={`${source}:${documentId}`}
						source={source}
						documentId={documentId}
						onOpenChunk={(chunkId) => setChunk({ source, chunkId })}
					/>
				) : (
					<p className="p-6 text-sm text-muted-foreground">
						Pick a document to see its content, outline and chunks.
					</p>
				)}
			</section>
			<ChunkDialog
				chunk={chunk}
				onOpen={setChunk}
				onClose={() => setChunk(null)}
			/>
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
