"use client";

import { GroundingPages } from "@/components/PageViewer";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import type { Citation } from "@/lib/chat";
import { withBasePath } from "@/lib/paths";

export function visualizeUrl(citation: Citation): string {
	const chunkIds = citation.chunk_ids?.length
		? citation.chunk_ids.join(",")
		: citation.chunk_id;
	const query = new URLSearchParams();
	if (citation.source) query.set("source", citation.source);
	if (citation.doc_item_refs?.length) {
		query.set("refs", JSON.stringify(citation.doc_item_refs));
	}
	return withBasePath(
		`/api/backend/api/visualize/${encodeURIComponent(chunkIds)}?${query}`,
	);
}

export default function VisualGroundingDialog({
	citation,
	onClose,
}: {
	citation: Citation | null;
	onClose: () => void;
}) {
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
				{citation && <GroundingPages url={visualizeUrl(citation)} />}
			</DialogContent>
		</Dialog>
	);
}
