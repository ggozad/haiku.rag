"use client";

import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { useJson } from "@/lib/useJson";

interface ChunkRow {
	id: string;
	document_id: string;
	order: number;
	content: string;
	metadata: Record<string, unknown>;
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

export default function ChunkDialog({
	source,
	chunkId,
	onClose,
}: {
	source: string;
	chunkId: string | null;
	onClose: () => void;
}) {
	const { data: chunk, error } = useJson<ChunkRow>(
		chunkId
			? `/api/backend/api/inspect/${encodeURIComponent(source)}/chunks/${encodeURIComponent(chunkId)}`
			: null,
	);
	const extra = chunk
		? Object.entries(chunk.metadata).filter(
				([key]) => !PROVENANCE.includes(key),
			)
		: [];

	return (
		<Dialog open={chunkId !== null} onOpenChange={(open) => !open && onClose()}>
			<DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-3xl">
				<DialogHeader>
					<DialogTitle>Chunk {chunk ? chunk.order : ""}</DialogTitle>
					<DialogDescription className="break-all font-mono text-xs">
						{chunkId}
					</DialogDescription>
				</DialogHeader>
				{error && <p className="text-sm text-destructive">{error}</p>}
				{chunk && (
					<div className="space-y-4">
						<pre className="whitespace-pre-wrap rounded-md bg-muted p-3 text-sm">
							{chunk.content}
						</pre>
						<dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
							{PROVENANCE.map((key) => (
								<Field key={key} label={key} value={chunk.metadata[key]} />
							))}
							{extra.map(([key, value]) => (
								<Field key={key} label={key} value={value} />
							))}
						</dl>
					</div>
				)}
			</DialogContent>
		</Dialog>
	);
}
