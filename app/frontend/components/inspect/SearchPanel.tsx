"use client";

import { ImagePlus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { type SearchInputs, searchHref } from "@/lib/addresses";
import type { DatabaseEntry } from "@/lib/databases";
import { decoded } from "@/lib/http";
import { type Attachment, readImage } from "@/lib/images";
import { firstLine, searched } from "@/lib/inspect";
import type { ChunkRef } from "./ChunkView";

interface Result {
	source: string;
	chunk_id: string;
	document_id: string;
	document_title: string | null;
	document_uri: string | null;
	score: number;
	page_numbers: number[];
	headings: string[] | null;
	content: string;
}

interface Answer {
	results: Result[];
	rendering: string;
}

// Each search's image and answer, by its address. An image cannot ride in a
// URL, so these last until the browser reloads.
// ponytail: one entry per search run in this tab, never evicted; cap it if that grows.
const held = new Map<
	string,
	{ image: Attachment | null; answer: Answer | null }
>();

// Runs the search its `inputs` name, once, as it mounts: its owner gives it a
// new key for each address.
export default function SearchPanel({
	databases,
	inputs,
	onSearch,
	onOpen,
}: {
	databases: DatabaseEntry[];
	inputs: SearchInputs;
	onSearch: (inputs: SearchInputs) => void;
	onOpen: (chunk: ChunkRef) => void;
}) {
	const key = searchHref(inputs);
	const entry = held.get(key);
	const [query, setQuery] = useState(inputs.q);
	const [image, setImage] = useState<Attachment | null>(entry?.image ?? null);
	const [limit, setLimit] = useState(inputs.limit);
	const [type, setType] = useState(inputs.type);
	const [filter, setFilter] = useState(inputs.filter);
	// An address naming databases picks those, and one naming none picks every
	// available database, those becoming available later included.
	const [picked, setPicked] = useState<Set<string> | null>(() =>
		inputs.dbs.length ? new Set(inputs.dbs) : null,
	);
	const [unticked, setUnticked] = useState<Set<string>>(() => new Set());
	const sources = picked
		? databases.map((db) => db.name).filter((name) => picked.has(name))
		: searched(databases, unticked);
	const ticked = (name: string) =>
		picked ? picked.has(name) : !unticked.has(name);
	const toggle = (name: string) => {
		const flip = (prev: Set<string>) => {
			const next = new Set(prev);
			if (next.has(name)) next.delete(name);
			else next.add(name);
			return next;
		};
		if (picked) setPicked(flip(picked));
		else setUnticked(flip);
	};
	const [answer, setAnswer] = useState<Answer>(
		entry?.answer ?? { results: [], rendering: "" },
	);
	const { results, rendering } = answer;
	const [asAgent, setAsAgent] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [busy, setBusy] = useState(false);
	const picker = useRef<HTMLInputElement>(null);
	const controller = useRef<AbortController | null>(null);
	const imageSearch = databases.some((db) => db.supports_image_search);

	const run = async (searchImage: Attachment | null) => {
		controller.current?.abort();
		const current = new AbortController();
		controller.current = current;
		setBusy(true);
		setError(null);
		try {
			const response = await fetch("/api/backend/api/search", {
				method: "POST",
				signal: current.signal,
				headers: { "content-type": "application/json" },
				body: JSON.stringify({
					...(searchImage ? { image: searchImage.data } : { query: inputs.q }),
					limit: Number(inputs.limit),
					search_type: searchImage ? "vector" : inputs.type,
					filter: inputs.filter.trim() || null,
					sources: inputs.dbs.length
						? inputs.dbs
						: searched(databases, new Set()),
				}),
			});
			const data = await decoded<Answer>(response);
			if (current.signal.aborted) return;
			held.set(key, { image: searchImage, answer: data });
			setAnswer(data);
		} catch (err) {
			if (!current.signal.aborted) {
				setError(err instanceof Error ? err.message : String(err));
				setAnswer({ results: [], rendering: "" });
			}
		} finally {
			if (!current.signal.aborted) setBusy(false);
		}
	};

	// An address naming no database searches every available one, so it waits
	// until the databases are known.
	const known = inputs.dbs.length > 0 || databases.length > 0;
	// biome-ignore lint/correctness/useExhaustiveDependencies: runs once per address
	useEffect(() => {
		if (!known) return;
		if (!entry?.answer && (entry?.image || inputs.q)) run(entry?.image ?? null);
		return () => controller.current?.abort();
	}, [known]);

	const submit = () => {
		const next = {
			q: image ? "" : query.trim(),
			dbs: sources,
			type,
			limit,
			filter: filter.trim(),
		};
		held.set(searchHref(next), { image, answer: null });
		if (searchHref(next) === key) run(image);
		else onSearch(next);
	};

	return (
		<div className="flex min-h-0 min-w-0 flex-1 flex-col">
			<form
				className="space-y-2 border-b p-4"
				onSubmit={(event) => {
					event.preventDefault();
					submit();
				}}
			>
				<div className="flex gap-2">
					{image ? (
						<div className="flex flex-1 items-center gap-2 rounded-md border px-2 text-sm">
							{/* biome-ignore lint/performance/noImgElement: local data URL preview */}
							<img
								src={`data:${image.mimeType};base64,${image.data}`}
								alt={image.name}
								className="size-8 rounded object-cover"
							/>
							<span className="truncate">{image.name}</span>
							<Button
								type="button"
								size="icon-xs"
								variant="ghost"
								aria-label="Remove the image"
								onClick={() => setImage(null)}
							>
								<X />
							</Button>
						</div>
					) : (
						<Input
							id="search-query"
							placeholder="Search the documents"
							value={query}
							onChange={(event) => setQuery(event.target.value)}
						/>
					)}
					{imageSearch && (
						<>
							<input
								ref={picker}
								id="search-image"
								type="file"
								accept="image/*"
								hidden
								onChange={async (event) => {
									const file = event.target.files?.[0];
									event.target.value = "";
									if (file) setImage(await readImage(file));
								}}
							/>
							<Button
								type="button"
								variant="outline"
								size="icon"
								aria-label="Search by image"
								onClick={() => picker.current?.click()}
							>
								<ImagePlus />
							</Button>
						</>
					)}
					<Button
						type="submit"
						disabled={busy || !sources.length || (!image && !query.trim())}
					>
						Search
					</Button>
				</div>
				<div className="flex flex-wrap items-center gap-2 text-sm">
					<Select value={type} onValueChange={setType} disabled={!!image}>
						<SelectTrigger className="w-32" aria-label="Search type">
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="hybrid">Hybrid</SelectItem>
							<SelectItem value="vector">Vector</SelectItem>
							<SelectItem value="fts">Full text</SelectItem>
						</SelectContent>
					</Select>
					<Select value={limit} onValueChange={setLimit}>
						<SelectTrigger className="w-24" aria-label="Results">
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							{["5", "10", "20", "50"].map((n) => (
								<SelectItem key={n} value={n}>
									{n} results
								</SelectItem>
							))}
						</SelectContent>
					</Select>
					<Input
						id="search-filter"
						className="w-80 font-mono text-xs"
						placeholder="SQL filter on documents"
						value={filter}
						onChange={(event) => setFilter(event.target.value)}
					/>
					{(databases.length > 1 || sources.length < databases.length) &&
						databases.map((db) => (
							<label
								key={db.name}
								htmlFor={`search-db-${db.name}`}
								className="flex items-center gap-1"
							>
								<Checkbox
									id={`search-db-${db.name}`}
									checked={ticked(db.name)}
									onCheckedChange={() => toggle(db.name)}
								/>
								{db.name}
							</label>
						))}
				</div>
				{error && <p className="text-sm text-destructive">{error}</p>}
			</form>
			<div className="min-h-0 flex-1 overflow-y-auto p-4">
				{results.length > 0 && (
					<div className="mb-3 flex gap-2">
						<Button
							size="sm"
							variant={asAgent ? "outline" : "secondary"}
							onClick={() => setAsAgent(false)}
						>
							Results
						</Button>
						<Button
							size="sm"
							variant={asAgent ? "secondary" : "outline"}
							onClick={() => setAsAgent(true)}
						>
							As the agent sees them
						</Button>
					</div>
				)}
				{asAgent ? (
					<pre className="whitespace-pre-wrap font-mono text-xs">
						{rendering}
					</pre>
				) : (
					<ol className="space-y-2">
						{results.map((result, rank) => (
							<li key={`${result.source}:${result.chunk_id}`}>
								<button
									type="button"
									className="w-full rounded-md border px-3 py-2 text-left hover:bg-muted"
									onClick={() =>
										onOpen({
											source: result.source,
											documentId: result.document_id,
											chunkId: result.chunk_id,
										})
									}
								>
									<div className="flex gap-2 text-xs text-muted-foreground">
										<span className="tabular-nums">#{rank + 1}</span>
										<span className="tabular-nums">
											{result.score.toFixed(3)}
										</span>
										<span>{result.source}</span>
										<span className="truncate">
											{result.document_title || result.document_uri}
										</span>
										{result.page_numbers.length > 0 && (
											<span>p. {result.page_numbers.join(", ")}</span>
										)}
									</div>
									{result.headings && (
										<div className="truncate text-xs text-muted-foreground">
											{result.headings.join(" › ")}
										</div>
									)}
									<div className="text-sm">
										{firstLine(result.content, 300)}
									</div>
								</button>
							</li>
						))}
					</ol>
				)}
			</div>
		</div>
	);
}
