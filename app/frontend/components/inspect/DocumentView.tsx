"use client";

import { useMemo } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { blocksOf, type Item, type OutlineNode } from "@/lib/inspect";
import { useJson } from "@/lib/useJson";
import ChunksTab, { type ChunkPage, PAGE } from "./ChunksTab";
import ContentTab from "./ContentTab";
import OutlineTab from "./OutlineTab";
import PagesTab, { type PageEntry } from "./PagesTab";
import PicturesTab, { type Picture } from "./PicturesTab";

interface DocumentInfo {
	id: string;
	title: string | null;
	uri: string | null;
	metadata: Record<string, unknown>;
	created_at: string;
	updated_at: string;
}

// The open tab, the chunk list's offset, the drawn page and the highlighted item.
export interface DocumentState {
	tab: string;
	offset: number;
	page: number | null;
	item: number | null;
}

export const OPENING: DocumentState = {
	tab: "content",
	offset: 0,
	page: null,
	item: null,
};

export default function DocumentView({
	source,
	documentId,
	state,
	onState,
	onOpenChunk,
	actions,
}: {
	source: string;
	documentId: string;
	state: DocumentState;
	onState: (change: Partial<DocumentState>) => void;
	onOpenChunk: (chunkId: string) => void;
	actions?: React.ReactNode;
}) {
	const { tab, offset, page, item } = state;
	const base = `/api/backend/api/inspect/${encodeURIComponent(source)}/documents/${encodeURIComponent(documentId)}`;
	const doc = useJson<DocumentInfo>(base);
	const items = useJson<{ items: Item[] }>(`${base}/items`);
	const outline = useJson<{ outline: OutlineNode[] }>(`${base}/outline`);
	const chunks = useJson<ChunkPage>(
		`${base}/chunks?limit=${PAGE}&offset=${offset}`,
	);
	const pages = useJson<{ pages: PageEntry[] }>(
		tab === "pages" ? `${base}/pages` : null,
	);
	const pictureList = useJson<{ pictures: Picture[] }>(
		tab === "pictures" ? `${base}/pictures` : null,
	);
	const blocks = useMemo(
		() => (items.data ? blocksOf(items.data.items) : []),
		[items.data],
	);

	const pictureUrl = (ref: string) =>
		`/api/backend/api/pictures?${new URLSearchParams({ source, document_id: documentId, ref })}`;

	return (
		<div className="flex min-h-0 min-w-0 flex-1 flex-col">
			<div className="space-y-1 border-b px-4 py-3">
				<h2 className="font-semibold">
					{doc.data?.title || doc.data?.uri || documentId}
				</h2>
				{doc.data?.uri && doc.data.uri !== doc.data.title && (
					<p className="break-all text-xs text-muted-foreground">
						{doc.data.uri}
					</p>
				)}
				{doc.error && <p className="text-sm text-destructive">{doc.error}</p>}
				{doc.data && actions}
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
				onValueChange={(value) => onState({ tab: value })}
				className="flex min-h-0 flex-1 flex-col"
			>
				<TabsList className="mx-4 mt-3">
					<TabsTrigger value="content">Content</TabsTrigger>
					<TabsTrigger value="outline">Outline</TabsTrigger>
					<TabsTrigger value="chunks">Chunks</TabsTrigger>
					<TabsTrigger value="pages">Pages</TabsTrigger>
					<TabsTrigger value="pictures">Pictures</TabsTrigger>
				</TabsList>
				<TabsContent
					value="content"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					<ContentTab
						items={items.data?.items}
						blocks={blocks}
						loading={items.loading}
						error={items.error}
						pictureUrl={pictureUrl}
						highlighted={item}
						onOpenChunk={onOpenChunk}
					/>
				</TabsContent>
				<TabsContent
					value="outline"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					<OutlineTab
						outline={outline.data?.outline}
						error={outline.error}
						onJump={(index) => onState({ tab: "content", item: index })}
					/>
				</TabsContent>
				<TabsContent
					value="chunks"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					<ChunksTab
						page={chunks.data}
						error={chunks.error}
						offset={offset}
						onOffset={(next) => onState({ offset: next })}
						onOpenChunk={onOpenChunk}
					/>
				</TabsContent>
				<TabsContent
					value="pages"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					<PagesTab
						pages={pages.data?.pages}
						error={pages.error}
						page={page}
						pageUrl={(n) => `${base}/pages/${n}`}
						onPage={(next) => onState({ page: next })}
						onOpenChunk={onOpenChunk}
					/>
				</TabsContent>
				<TabsContent
					value="pictures"
					className="min-h-0 flex-1 overflow-y-auto px-4 py-3"
				>
					<PicturesTab
						pictures={pictureList.data?.pictures}
						error={pictureList.error}
						pictureUrl={pictureUrl}
						onOpenChunk={onOpenChunk}
					/>
				</TabsContent>
			</Tabs>
		</div>
	);
}
