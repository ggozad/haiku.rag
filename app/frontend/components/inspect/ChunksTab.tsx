"use client";

import { Button } from "@/components/ui/button";
import { firstLine } from "@/lib/inspect";

export const PAGE = 50;

export interface ChunkPage {
	total: number;
	chunks: {
		id: string;
		order: number;
		content: string;
		metadata: { page_numbers?: number[]; labels?: string[] };
	}[];
}

export default function ChunksTab({
	page,
	error,
	offset,
	onOffset,
	onOpenChunk,
}: {
	page: ChunkPage | null;
	error: string | null;
	offset: number;
	onOffset: (offset: number) => void;
	onOpenChunk: (chunkId: string) => void;
}) {
	return (
		<>
			{error && <p className="text-sm text-destructive">{error}</p>}
			<ul className="space-y-1">
				{page?.chunks.map((chunk) => (
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
			{page && page.total > PAGE && (
				<div className="mt-3 flex items-center gap-2 text-sm">
					<Button
						size="sm"
						variant="outline"
						disabled={offset === 0}
						onClick={() => onOffset(Math.max(0, offset - PAGE))}
					>
						Previous
					</Button>
					<span className="text-muted-foreground tabular-nums">
						{offset + 1}–{offset + page.chunks.length} of {page.total}
					</span>
					<Button
						size="sm"
						variant="outline"
						disabled={offset + PAGE >= page.total}
						onClick={() => onOffset(offset + PAGE)}
					>
						Next
					</Button>
				</div>
			)}
		</>
	);
}
