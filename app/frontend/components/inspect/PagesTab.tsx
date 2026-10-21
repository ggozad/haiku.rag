"use client";

import { Button } from "@/components/ui/button";

export interface PageEntry {
	page: number;
	chunk_ids: string[];
}

// The pages any item covers, and one page drawn with its chunks.
export default function PagesTab({
	pages,
	error,
	page,
	pageUrl,
	onPage,
	onOpenChunk,
}: {
	pages: PageEntry[] | undefined;
	error: string | null;
	page: number | null;
	pageUrl: (page: number) => string;
	onPage: (page: number) => void;
	onOpenChunk: (chunkId: string) => void;
}) {
	return (
		<>
			{error && <p className="text-sm text-destructive">{error}</p>}
			{pages && pages.length === 0 && (
				<p className="text-sm text-muted-foreground">
					No chunk of this document has a place on a page.
				</p>
			)}
			<div className="flex flex-wrap gap-1">
				{pages?.map((entry) => (
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
			{page !== null && (
				<div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_14rem]">
					{/* biome-ignore lint/performance/noImgElement: page render from the backend */}
					<img
						src={pageUrl(page)}
						alt={`Page ${page} with its chunks drawn`}
						className="w-full rounded border"
					/>
					<ul className="space-y-1 text-sm">
						{pages
							?.find((entry) => entry.page === page)
							?.chunk_ids.map((chunkId) => (
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
			)}
		</>
	);
}
