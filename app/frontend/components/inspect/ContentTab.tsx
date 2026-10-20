"use client";

import { useEffect } from "react";
import type { Block, Item } from "@/lib/inspect";
import { cn } from "@/lib/utils";

const TINTS = ["bg-sky-500/10", "bg-amber-500/10"];

function ItemBlock({
	item,
	block,
	pictureUrl,
	onOpen,
	index,
	highlighted,
}: {
	item: Item;
	block: Block;
	pictureUrl: string;
	onOpen: (chunkId: string) => void;
	index: number;
	highlighted: boolean;
}) {
	const className = cn(
		"rounded px-2 py-1 text-left",
		block.tint !== null && TINTS[block.tint],
		highlighted && "ring-2 ring-primary",
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
	const current = highlighted ? "location" : undefined;
	const title = item.chunk_ids.length
		? `Chunks: ${item.chunk_ids.join(", ")}`
		: "In no chunk";
	return block.chunkId ? (
		<button
			type="button"
			id={`item-${index}`}
			className={cn(className, "block w-full")}
			title={title}
			aria-current={current}
			onClick={() => block.chunkId && onOpen(block.chunkId)}
		>
			{body}
		</button>
	) : (
		<div
			id={`item-${index}`}
			className={className}
			title={title}
			aria-current={current}
		>
			{body}
		</div>
	);
}

// The document's items in reading order, tinted by the chunk holding them.
// The highlighted item is scrolled to once this tab has rendered it.
export default function ContentTab({
	items,
	blocks,
	loading,
	error,
	pictureUrl,
	highlighted,
	onOpenChunk,
}: {
	items: Item[] | undefined;
	blocks: Block[];
	loading: boolean;
	error: string | null;
	pictureUrl: (ref: string) => string;
	highlighted: number | null;
	onOpenChunk: (chunkId: string) => void;
}) {
	// biome-ignore lint/correctness/useExhaustiveDependencies: blocks render the targets
	useEffect(() => {
		if (highlighted === null) return;
		document
			.getElementById(`item-${highlighted}`)
			?.scrollIntoView({ block: "start" });
	}, [highlighted, blocks]);

	return (
		<>
			{loading && <p className="text-sm text-muted-foreground">Loading…</p>}
			{error && <p className="text-sm text-destructive">{error}</p>}
			<div className="space-y-1">
				{items?.map((item, index) => (
					<ItemBlock
						key={item.self_ref}
						item={item}
						block={blocks[index]}
						index={index}
						highlighted={index === highlighted}
						pictureUrl={pictureUrl(item.self_ref)}
						onOpen={onOpenChunk}
					/>
				))}
			</div>
		</>
	);
}
