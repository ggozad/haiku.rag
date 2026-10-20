"use client";

import { chunkCount, type OutlineNode } from "@/lib/inspect";

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

export default function OutlineTab({
	outline,
	error,
	onJump,
}: {
	outline: OutlineNode[] | undefined;
	error: string | null;
	onJump: (index: number) => void;
}) {
	return (
		<>
			{error && <p className="text-sm text-destructive">{error}</p>}
			{outline && outline.length === 0 && (
				<p className="text-sm text-muted-foreground">
					This document has no headings.
				</p>
			)}
			{outline && <OutlineTree nodes={outline} onJump={onJump} />}
		</>
	);
}
