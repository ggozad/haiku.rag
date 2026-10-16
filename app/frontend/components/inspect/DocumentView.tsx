"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
	type Block,
	blocksOf,
	chunkCount,
	firstLine,
	type Item,
	type OutlineNode,
} from "@/lib/inspect";
import { useJson } from "@/lib/useJson";
import { cn } from "@/lib/utils";

interface DocumentInfo {
	id: string;
	title: string | null;
	uri: string | null;
	metadata: Record<string, unknown>;
	created_at: string;
	updated_at: string;
}

interface ChunkPage {
	total: number;
	chunks: {
		id: string;
		order: number;
		content: string;
		metadata: { page_numbers?: number[]; labels?: string[] };
	}[];
}

const TINTS = ["bg-sky-500/10", "bg-amber-500/10"];
const PAGE = 50;

function ItemBlock({
	item,
	block,
	pictureUrl,
	onOpen,
	index,
}: {
	item: Item;
	block: Block;
	pictureUrl: string;
	onOpen: (chunkId: string) => void;
	index: number;
}) {
	const className = cn(
		"rounded px-2 py-1 text-left",
		block.tint !== null && TINTS[block.tint],
		block.chunkId && "cursor-pointer hover:ring-1 hover:ring-ring",
	);
	let body: React.ReactNode;
	switch (block.kind) {
		case "heading":
			body = (
				<span
					className={cn(
						"font-semibold",
						block.level <= 1 ? "text-lg" : block.level === 2 ? "text-base" : "",
					)}
				>
					{item.text}
				</span>
			);
			break;
		case "list":
			body = <span className="pl-4">• {item.text}</span>;
			break;
		case "caption":
			body = (
				<span className="text-xs italic text-muted-foreground">
					{item.text}
				</span>
			);
			break;
		case "table":
		case "code":
			body = (
				<pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs">
					{item.text}
				</pre>
			);
			break;
		case "picture":
			body = (
				// biome-ignore lint/performance/noImgElement: pictures come from the backend proxy
				<img
					src={pictureUrl}
					alt={item.text || `Picture ${item.self_ref}`}
					className="max-h-80 rounded border"
				/>
			);
			break;
		default:
			body = <span className="whitespace-pre-wrap">{item.text}</span>;
	}
	const title = item.chunk_ids.length
		? `Chunks: ${item.chunk_ids.join(", ")}`
		: "In no chunk";
	return block.chunkId ? (
		<button
			type="button"
			id={`item-${index}`}
			className={cn(className, "block w-full")}
			title={title}
			onClick={() => block.chunkId && onOpen(block.chunkId)}
		>
			{body}
		</button>
	) : (
		<div id={`item-${index}`} className={className} title={title}>
			{body}
		</div>
	);
}

function OutlineTree({
	nodes,
	onJump,
}: {
	nodes: OutlineNode[];
	onJump: (index: number) => void;
}) {
	return (
		<ul className="space-y-1 border-l pl-3">
			{nodes.map((node) => (
				<li key={node.self_ref}>
					<button
						type="button"
						className="flex w-full items-baseline gap-2 text-left text-sm hover:underline"
						onClick={() => onJump(node.item_range[0])}
					>
						<span>{node.title}</span>
						<span className="text-xs text-muted-foreground">
							{node.page_numbers.length ? `p. ${node.page_numbers[0]} · ` : ""}
							{chunkCount(node)} chunks
						</span>
					</button>
					{node.children.length > 0 && (
						<OutlineTree nodes={node.children} onJump={onJump} />
					)}
				</li>
			))}
		</ul>
	);
}

export default function DocumentView({
	source,
	documentId,
	onOpenChunk,
}: {
	source: string;
	documentId: string;
	onOpenChunk: (chunkId: string) => void;
}) {
	const base = `/api/backend/api/inspect/${encodeURIComponent(source)}/documents/${encodeURIComponent(documentId)}`;
	const [tab, setTab] = useState("content");
	const [jumpTo, setJumpTo] = useState<number | null>(null);
	const [offset, setOffset] = useState(0);
	const doc = useJson<DocumentInfo>(base);
	const items = useJson<{ items: Item[] }>(`${base}/items`);
	const outline = useJson<{ outline: OutlineNode[] }>(`${base}/outline`);
	const chunks = useJson<ChunkPage>(
		`${base}/chunks?limit=${PAGE}&offset=${offset}`,
	);
	const blocks = useMemo(
		() => (items.data ? blocksOf(items.data.items) : []),
		[items.data],
	);

	// A jump waits until its item is on the page.
	// biome-ignore lint/correctness/useExhaustiveDependencies: blocks render the targets
	useEffect(() => {
		if (jumpTo === null || tab !== "content") return;
		const target = document.getElementById(`item-${jumpTo}`);
		if (!target) return;
		target.scrollIntoView({ block: "start" });
		setJumpTo(null);
	}, [jumpTo, tab, blocks]);

	const pictureUrl = (ref: string) =>
		`/api/backend/api/pictures?${new URLSearchParams({ source, document_id: documentId, ref })}`;

	return (
		<div className="flex h-full min-w-0 flex-col">
			<div className="space-y-1 border-b px-4 py-3">
				<h2 className="font-semibold">
					{doc.data?.title || doc.data?.uri || documentId}
				</h2>
				{doc.data?.uri && (
					<p className="break-all text-xs text-muted-foreground">
						{doc.data.uri}
					</p>
				)}
				{doc.error && <p className="text-sm text-destructive">{doc.error}</p>}
				{doc.data && (
					<details className="text-xs">
						<summary className="cursor-pointer text-muted-foreground">
							Metadata
						</summary>
						<dl className="mt-1 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4">
							{Object.entries({
								created: doc.data.created_at,
								updated: doc.data.updated_at,
								...doc.data.metadata,
							}).map(([key, value]) => (
								<div key={key} className="contents">
									<dt className="text-muted-foreground">{key}</dt>
									<dd className="break-all font-mono">
										{typeof value === "object"
											? JSON.stringify(value)
											: String(value)}
									</dd>
								</div>
							))}
						</dl>
					</details>
				)}
			</div>
			<Tabs
				value={tab}
				onValueChange={(value) => {
					setJumpTo(null);
					setTab(value);
				}}
				className="flex min-h-0 flex-1 flex-col"
			>
				<TabsList className="mx-4 mt-3">
					<TabsTrigger value="content">Content</TabsTrigger>
					<TabsTrigger value="outline">Outline</TabsTrigger>
					<TabsTrigger value="chunks">Chunks</TabsTrigger>
				</TabsList>
				<TabsContent
					value="content"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					{items.loading && (
						<p className="text-sm text-muted-foreground">Loading…</p>
					)}
					{items.error && (
						<p className="text-sm text-destructive">{items.error}</p>
					)}
					<div className="space-y-1">
						{items.data?.items.map((item, index) => (
							<ItemBlock
								key={item.self_ref}
								item={item}
								block={blocks[index]}
								index={index}
								pictureUrl={pictureUrl(item.self_ref)}
								onOpen={onOpenChunk}
							/>
						))}
					</div>
				</TabsContent>
				<TabsContent
					value="outline"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					{outline.error && (
						<p className="text-sm text-destructive">{outline.error}</p>
					)}
					{outline.data && outline.data.outline.length === 0 && (
						<p className="text-sm text-muted-foreground">
							This document has no headings.
						</p>
					)}
					{outline.data && (
						<OutlineTree
							nodes={outline.data.outline}
							onJump={(index) => {
								setTab("content");
								setJumpTo(index);
							}}
						/>
					)}
				</TabsContent>
				<TabsContent
					value="chunks"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					{chunks.error && (
						<p className="text-sm text-destructive">{chunks.error}</p>
					)}
					<ul className="space-y-1">
						{chunks.data?.chunks.map((chunk) => (
							<li key={chunk.id}>
								<button
									type="button"
									className="flex w-full gap-3 rounded px-2 py-1 text-left text-sm hover:bg-muted"
									onClick={() => onOpenChunk(chunk.id)}
								>
									<span className="w-10 shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
										{chunk.order}
									</span>
									<span className="min-w-0 flex-1 truncate">
										{firstLine(chunk.content)}
									</span>
									<span className="shrink-0 text-xs text-muted-foreground">
										{[
											chunk.metadata.page_numbers?.length
												? `p. ${chunk.metadata.page_numbers.join(", ")}`
												: "",
											chunk.metadata.labels?.join(", ") ?? "",
										]
											.filter(Boolean)
											.join(" · ")}
									</span>
								</button>
							</li>
						))}
					</ul>
					{chunks.data && chunks.data.total > PAGE && (
						<div className="mt-3 flex items-center gap-2 text-sm">
							<Button
								size="sm"
								variant="outline"
								disabled={offset === 0}
								onClick={() => setOffset(Math.max(0, offset - PAGE))}
							>
								Previous
							</Button>
							<span className="text-muted-foreground tabular-nums">
								{offset + 1}–{offset + chunks.data.chunks.length} of{" "}
								{chunks.data.total}
							</span>
							<Button
								size="sm"
								variant="outline"
								disabled={offset + PAGE >= chunks.data.total}
								onClick={() => setOffset(offset + PAGE)}
							>
								Next
							</Button>
						</div>
					)}
				</TabsContent>
			</Tabs>
		</div>
	);
}
