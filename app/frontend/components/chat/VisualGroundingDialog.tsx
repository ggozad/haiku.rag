"use client";

import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import type { Citation } from "@/lib/chat";
import { useJson } from "@/lib/useJson";

export function visualizeUrl(citation: Citation): string {
	const chunkIds = citation.chunk_ids?.length
		? citation.chunk_ids.join(",")
		: citation.chunk_id;
	const query = new URLSearchParams();
	if (citation.source) query.set("source", citation.source);
	if (citation.doc_item_refs?.length) {
		query.set("refs", JSON.stringify(citation.doc_item_refs));
	}
	return `/api/backend/api/visualize/${encodeURIComponent(chunkIds)}?${query}`;
}

export default function VisualGroundingDialog({
	citation,
	onClose,
}: {
	citation: Citation | null;
	onClose: () => void;
}) {
	const grounding = useJson<{ images?: string[]; message?: string | null }>(
		citation ? visualizeUrl(citation) : null,
	);
	const images = grounding.data?.images ?? [];
	const message = grounding.error ?? grounding.data?.message ?? null;
	const loading = grounding.loading;

	return (
		<Dialog
			open={citation !== null}
			onOpenChange={(open) => !open && onClose()}
		>
			<DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-4xl">
				<DialogHeader>
					<DialogTitle>
						{citation?.document_title || citation?.document_uri}
					</DialogTitle>
					<DialogDescription>
						The cited passage highlighted on its pages.
					</DialogDescription>
				</DialogHeader>
				{loading && <p className="text-sm text-muted-foreground">Loading…</p>}
				{message && <p className="text-sm text-muted-foreground">{message}</p>}
				<div className="space-y-4">
					{images.map((image, n) => (
						// biome-ignore lint/performance/noImgElement: base64 page renders
						<img
							// biome-ignore lint/suspicious/noArrayIndexKey: pages have no id
							key={n}
							src={`data:image/png;base64,${image}`}
							alt={`Page ${n + 1} of the cited passage`}
							className="w-full rounded border"
						/>
					))}
				</div>
			</DialogContent>
		</Dialog>
	);
}
