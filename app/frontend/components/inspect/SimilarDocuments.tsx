"use client";

import { ChevronDown, Network } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Estimate, useEstimate } from "@/components/doctor/Estimate";
import {
	doctorBase,
	SimilarityControls,
	useSimilarityJob,
} from "@/components/doctor/SimilarityJob";
import { Button } from "@/components/ui/button";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { nearestDocuments } from "@/lib/doctor";
import { fileName } from "@/lib/ingestion";
import { useJson } from "@/lib/useJson";

const SHOWN = 10;

// The documents nearest this one by centroid, folded behind one button, or why
// there are none and the controls to compute them.
// ponytail: reads the whole job to find one document. Add a membership route if corpora grow large.
export default function SimilarDocuments({
	source,
	documentId,
	hrefOf,
}: {
	source: string;
	documentId: string;
	hrefOf: (id: string) => string;
}) {
	const [open, setOpen] = useState(false);
	const similarity = useSimilarityJob(source);
	const { result, stale } = similarity;
	const compared = result?.documents.some((doc) => doc.id === documentId);
	const listed = open && compared && result?.neighbours;
	const scores = useJson<{ generation: string; similarity: number[] }>(
		listed
			? `${doctorBase(source)}/similarity/documents?${new URLSearchParams({ id: documentId, generation: result.generation })}`
			: null,
	);
	const current = listed && stale === false;
	const estimate = useEstimate(open && !current ? source : null);
	if (!similarity.read)
		return similarity.error ? (
			<p className="text-sm text-destructive">{similarity.error}</p>
		) : null;
	const nearest =
		result &&
		nearestDocuments(
			scores.data,
			result.generation,
			result.documents,
			documentId,
			SHOWN,
		);
	return (
		<Collapsible open={open} onOpenChange={setOpen} className="contents">
			<CollapsibleTrigger asChild>
				<Button
					size="sm"
					variant="ghost"
					className="group text-muted-foreground"
				>
					<Network />
					Similar documents
					<ChevronDown className="transition-transform group-data-[state=open]:rotate-180" />
				</Button>
			</CollapsibleTrigger>
			<CollapsibleContent className="basis-full space-y-1 text-sm">
				{listed ? (
					<>
						{scores.error && <p className="text-destructive">{scores.error}</p>}
						<ul className="max-w-xl">
							{nearest?.map((doc) => (
								<li key={doc.id}>
									<Link
										href={hrefOf(doc.id)}
										title={doc.uri ?? undefined}
										className="flex w-full justify-between gap-2 py-0.5 hover:underline"
									>
										<span className="truncate">
											{doc.title || (doc.uri ? fileName(doc.uri) : doc.id)}
										</span>
										<span className="shrink-0 font-mono text-xs tabular-nums">
											{doc.similarity.toFixed(3)}
										</span>
									</Link>
								</li>
							))}
						</ul>
						{!current && (
							<>
								<Estimate text={estimate.text} />
								<SimilarityControls
									similarity={similarity}
									ready={estimate.ready}
								/>
							</>
						)}
					</>
				) : (
					<>
						<p className="text-muted-foreground">
							{!result
								? "Similarity has not been computed for this database."
								: compared
									? "The compared centroids are no longer kept in memory."
									: stale === false
										? "This document has no usable embedding."
										: stale
											? "This document was not compared: the database changed since similarity was computed."
											: "This document was not compared."}
						</p>
						<Estimate text={estimate.text} />
						<SimilarityControls
							similarity={similarity}
							ready={estimate.ready}
						/>
					</>
				)}
			</CollapsibleContent>
		</Collapsible>
	);
}
