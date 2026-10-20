"use client";

import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { firstLine } from "@/lib/inspect";
import { useJson } from "@/lib/useJson";

export interface ChunkRef {
	source: string;
	documentId: string;
	chunkId: string;
}

interface ChunkRow {
	id: string;
	document_id: string;
	order: number;
	content: string;
	metadata: Record<string, unknown>;
}

interface Context {
	rendering: string;
	pictures: string[];
	document_id: string;
}

interface Similar {
	results: {
		source: string;
		chunk_id: string;
		document_id: string;
		document_title: string | null;
		document_uri: string | null;
		score: number;
		content: string;
	}[];
}

const PROVENANCE = ["page_numbers", "headings", "labels", "doc_item_refs"];

function Field({ label, value }: { label: string; value: unknown }) {
	if (value === undefined || value === null) return null;
	if (Array.isArray(value) && value.length === 0) return null;
	return (
		<div className="contents">
			<dt className="text-muted-foreground">{label}</dt>
			<dd className="break-all font-mono text-xs">
				{Array.isArray(value)
					? value.join(label === "headings" ? " › " : ", ")
					: typeof value === "object"
						? JSON.stringify(value)
						: String(value)}
			</dd>
		</div>
	);
}

function Loading({ error }: { error: string | null }) {
	return error ? (
		<p className="text-sm text-destructive">{error}</p>
	) : (
		<p className="text-sm text-muted-foreground">Loading…</p>
	);
}

export const VIEWS = ["details", "agent", "grounding", "similar"];

export default function ChunkDialog({
	chunk: ref,
	view,
	onView,
	onOpen,
	onClose,
}: {
	chunk: ChunkRef | null;
	view: string;
	onView: (view: string) => void;
	onOpen: (chunk: ChunkRef) => void;
	onClose: () => void;
}) {
	const tab = VIEWS.includes(view) ? view : "details";
	const base = ref
		? `/api/backend/api/inspect/${encodeURIComponent(ref.source)}/chunks/${encodeURIComponent(ref.chunkId)}`
		: null;
	const chunk = useJson<ChunkRow>(base);
	const context = useJson<Context>(
		base && tab === "agent" ? `${base}/context` : null,
	);
	const grounding = useJson<{ images: string[]; message: string | null }>(
		ref && tab === "grounding"
			? `/api/backend/api/visualize/${encodeURIComponent(ref.chunkId)}?source=${encodeURIComponent(ref.source)}`
			: null,
	);
	const similar = useJson<Similar>(
		base && tab === "similar" ? `${base}/similar` : null,
	);
	const extra = chunk.data
		? Object.entries(chunk.data.metadata).filter(
				([key]) => !PROVENANCE.includes(key),
			)
		: [];
	const pictureUrl = (ref_: string) =>
		ref && context.data
			? `/api/backend/api/pictures?${new URLSearchParams({ source: ref.source, document_id: context.data.document_id, ref: ref_ })}`
			: "";

	return (
		<Dialog
			open={ref !== null}
			onOpenChange={(open) => {
				if (!open) onClose();
			}}
		>
			<DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-4xl">
				<DialogHeader>
					<DialogTitle>
						Chunk {chunk.data ? chunk.data.order : ""} · {ref?.source}
					</DialogTitle>
					<DialogDescription className="break-all font-mono text-xs">
						{ref?.chunkId}
					</DialogDescription>
				</DialogHeader>
				<Tabs value={tab} onValueChange={onView}>
					<TabsList>
						<TabsTrigger value="details">Details</TabsTrigger>
						<TabsTrigger value="agent">As the agent sees it</TabsTrigger>
						<TabsTrigger value="grounding">On the page</TabsTrigger>
						<TabsTrigger value="similar">Similar</TabsTrigger>
					</TabsList>
					<TabsContent value="details" className="space-y-4">
						{!chunk.data && <Loading error={chunk.error} />}
						{chunk.data && (
							<>
								<pre className="whitespace-pre-wrap rounded-md bg-muted p-3 text-sm">
									{chunk.data.content}
								</pre>
								<dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
									{PROVENANCE.map((key) => (
										<Field
											key={key}
											label={key}
											value={chunk.data?.metadata[key]}
										/>
									))}
									{extra.map(([key, value]) => (
										<Field key={key} label={key} value={value} />
									))}
								</dl>
							</>
						)}
					</TabsContent>
					<TabsContent value="agent" className="space-y-3">
						{!context.data && <Loading error={context.error} />}
						{context.data && (
							<>
								<pre className="whitespace-pre-wrap rounded-md bg-muted p-3 font-mono text-xs">
									{context.data.rendering}
								</pre>
								{context.data.pictures.length > 0 && (
									<div className="space-y-1">
										<p className="text-xs text-muted-foreground">
											Pictures a vision model receives with it
										</p>
										<div className="flex flex-wrap gap-2">
											{context.data.pictures.map((picture) => (
												// biome-ignore lint/performance/noImgElement: pictures come from the backend proxy
												<img
													key={picture}
													src={pictureUrl(picture)}
													alt={`Figure ${picture} of the document`}
													className="max-h-48 rounded border"
												/>
											))}
										</div>
									</div>
								)}
							</>
						)}
					</TabsContent>
					<TabsContent value="grounding" className="space-y-3">
						{!grounding.data && <Loading error={grounding.error} />}
						{grounding.data?.message && (
							<p className="text-sm text-muted-foreground">
								{grounding.data.message}
							</p>
						)}
						{grounding.data?.images.map((image, n) => (
							// biome-ignore lint/performance/noImgElement: base64 page renders
							<img
								// biome-ignore lint/suspicious/noArrayIndexKey: pages have no id
								key={n}
								src={`data:image/png;base64,${image}`}
								alt={`Page ${n + 1} of the chunk`}
								className="w-full rounded border"
							/>
						))}
					</TabsContent>
					<TabsContent value="similar">
						{!similar.data && <Loading error={similar.error} />}
						<ul className="space-y-1">
							{similar.data?.results.map((row) => (
								<li key={`${row.source}:${row.chunk_id}`}>
									<button
										type="button"
										className="w-full rounded px-2 py-1 text-left text-sm hover:bg-muted"
										onClick={() =>
											onOpen({
												source: row.source,
												documentId: row.document_id,
												chunkId: row.chunk_id,
											})
										}
									>
										<div className="flex gap-2 text-xs text-muted-foreground">
											<span className="tabular-nums">
												{row.score.toFixed(3)}
											</span>
											<span className="truncate">
												{row.document_title || row.document_uri}
											</span>
										</div>
										<div className="truncate">{firstLine(row.content)}</div>
									</button>
								</li>
							))}
						</ul>
						{similar.data && similar.data.results.length === 0 && (
							<p className="text-sm text-muted-foreground">
								Nothing else matched.
							</p>
						)}
					</TabsContent>
				</Tabs>
			</DialogContent>
		</Dialog>
	);
}
