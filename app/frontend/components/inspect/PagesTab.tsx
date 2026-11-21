"use client";

import { useEffect } from "react";
import PageViewer from "@/components/PageViewer";
import { Button } from "@/components/ui/button";
import {
	type Grounding,
	type Original,
	type PageSize,
	pageBoxesUrl,
	withBoxes,
} from "@/lib/grounding";
import { useJson } from "@/lib/useJson";

export interface PageEntry {
	page: number;
	chunk_ids: string[];
}

// The document's pages from its original, and on the selected page the boxes
// of every chunk-held item on it, with those chunks listed beside.
export default function PagesTab({
	source,
	documentId,
	original,
	pages,
	entries,
	error,
	page,
	reads,
	onPage,
	onOpenChunk,
	onStale,
}: {
	source: string;
	documentId: string;
	original: Original | null | undefined;
	pages: PageSize[];
	entries: PageEntry[] | undefined;
	error: string | null;
	page: number | null;
	reads: number;
	onPage: (page: number) => void;
	onOpenChunk: (chunkId: string) => void;
	onStale: () => void;
}) {
	const entry = entries?.find((candidate) => candidate.page === page);
	const grounding = useJson<Grounding>(
		entry?.chunk_ids.length
			? pageBoxesUrl(source, documentId, entry.page)
			: null,
		reads,
	);
	const drawnOn = grounding.data?.original?.md5;
	// The document read names another original than the boxes: read both again.
	// biome-ignore lint/correctness/useExhaustiveDependencies: fires once per disagreeing pair
	useEffect(() => {
		if (drawnOn && original && drawnOn !== original.md5) onStale();
	}, [drawnOn, original?.md5]);
	if (original === undefined) return null;
	if (original === null)
		return (
			<p className="text-sm text-muted-foreground">
				This document has no stored original to show its pages from.
			</p>
		);
	// Boxes read from another version of the original would land anywhere.
	const boxes =
		drawnOn === original.md5 && grounding.data
			? grounding.data.pages.filter((drawn) => drawn.page === page)
			: [];

	return (
		<>
			{(error || grounding.error) && (
				<p className="text-sm text-destructive">{error || grounding.error}</p>
			)}
			<div className="flex flex-wrap gap-1">
				{pages.map((entry) => (
					<Button
						key={entry.page}
						size="sm"
						variant={entry.page === page ? "secondary" : "outline"}
						onClick={() => onPage(entry.page)}
					>
						{entry.page}
					</Button>
				))}
			</div>
			<div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_14rem]">
				<PageViewer
					source={source}
					documentId={documentId}
					original={original}
					pages={withBoxes(pages, boxes)}
					focus={page}
					onStale={onStale}
				/>
				<ul className="space-y-1 self-start text-sm lg:sticky lg:top-0">
					{entry?.chunk_ids.map((chunkId) => (
						<li key={chunkId}>
							<button
								type="button"
								className="w-full truncate rounded px-2 py-1 text-left font-mono text-xs hover:bg-muted"
								onClick={() => onOpenChunk(chunkId)}
							>
								{chunkId}
							</button>
						</li>
					))}
				</ul>
			</div>
		</>
	);
}
