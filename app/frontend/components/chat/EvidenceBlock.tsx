"use client";

import { ChevronRight, FileText } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { Citation, Evidence } from "@/lib/chat";
import VisualGroundingDialog from "./VisualGroundingDialog";

export function pictureUrl(citation: Citation, ref: string): string {
	const query = new URLSearchParams({
		document_id: citation.document_id,
		ref,
	});
	if (citation.source) query.set("source", citation.source);
	return `/api/backend/api/pictures?${query}`;
}

function CitationItem({
	citation,
	showSource,
	onView,
}: {
	citation: Citation;
	showSource: boolean;
	onView: (citation: Citation) => void;
}) {
	const pages = citation.page_numbers.length
		? `p. ${citation.page_numbers.join(", ")}`
		: null;
	return (
		<Collapsible className="rounded-md border">
			<CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left text-sm">
				<ChevronRight className="size-3.5 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-90" />
				<span className="font-mono text-xs text-muted-foreground">
					[{citation.index}]
				</span>
				<span className="min-w-0 flex-1 truncate">
					{citation.document_title || citation.document_uri}
				</span>
				{showSource && citation.source && (
					<Badge variant="outline">{citation.source}</Badge>
				)}
				{pages && (
					<span className="shrink-0 text-xs text-muted-foreground">
						{pages}
					</span>
				)}
			</CollapsibleTrigger>
			<CollapsibleContent className="space-y-3 border-t px-3 py-3">
				{citation.headings && citation.headings.length > 0 && (
					<p className="text-xs text-muted-foreground">
						{citation.headings.join(" › ")}
					</p>
				)}
				<p className="whitespace-pre-wrap text-sm">{citation.content}</p>
				{citation.picture_refs && citation.picture_refs.length > 0 && (
					<div className="flex flex-wrap gap-2">
						{citation.picture_refs.map((ref) => (
							// biome-ignore lint/performance/noImgElement: pictures come from the backend proxy
							<img
								key={ref}
								src={pictureUrl(citation, ref)}
								alt={`Figure ${ref} from the cited document`}
								className="max-h-48 rounded border"
							/>
						))}
					</div>
				)}
				<Button variant="outline" size="sm" onClick={() => onView(citation)}>
					<FileText />
					View in document
				</Button>
			</CollapsibleContent>
		</Collapsible>
	);
}

export default function EvidenceBlock({
	evidence,
	showSource,
}: {
	evidence: Evidence;
	showSource: boolean;
}) {
	const [viewing, setViewing] = useState<Citation | null>(null);
	if (!evidence.citations.length && !evidence.program) return null;

	return (
		<div className="space-y-2">
			{evidence.citations.length > 0 && (
				<>
					<p className="text-xs font-medium text-muted-foreground">
						Sources ({evidence.citations.length})
					</p>
					{evidence.citations.map((citation) => (
						<CitationItem
							key={`${citation.source ?? ""}:${citation.chunk_id}`}
							citation={citation}
							showSource={showSource}
							onView={setViewing}
						/>
					))}
				</>
			)}
			{evidence.program && (
				<Collapsible className="rounded-md border">
					<CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left text-sm">
						<ChevronRight className="size-3.5 text-muted-foreground transition-transform group-data-[state=open]:rotate-90" />
						Program behind the answer
					</CollapsibleTrigger>
					<CollapsibleContent className="border-t">
						<pre className="overflow-x-auto p-3 font-mono text-xs">
							{evidence.program}
						</pre>
					</CollapsibleContent>
				</Collapsible>
			)}
			<VisualGroundingDialog
				citation={viewing}
				onClose={() => setViewing(null)}
			/>
		</div>
	);
}
