"use client";

import { ChevronDown, Map as MapIcon } from "lucide-react";
import Link from "next/link";
import {
	Estimate,
	MapControls,
	useEstimate,
	useMapJob,
} from "@/components/doctor/MapJob";
import { Button } from "@/components/ui/button";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";

// A document's place on its database's map, or, folded behind one button, why
// it has none and the controls to compute the map.
// ponytail: reads the whole map to find one document; add a membership route if maps grow large.
export default function ShowOnMap({
	source,
	documentId,
	href,
}: {
	source: string;
	documentId: string;
	href: string;
}) {
	const map = useMapJob(source);
	const { result } = map;
	const onMap = result?.points.some((point) => point.id === documentId);
	const estimate = useEstimate(map.read && !onMap ? source : null);
	if (!map.read)
		return map.error ? (
			<p className="text-sm text-destructive">{map.error}</p>
		) : null;
	if (onMap)
		return (
			<Button asChild size="sm" variant="outline">
				<Link href={href}>
					<MapIcon />
					Show on map
				</Link>
			</Button>
		);
	return (
		<Collapsible className="contents">
			<CollapsibleTrigger asChild>
				<Button
					size="sm"
					variant="ghost"
					className="group text-muted-foreground"
				>
					<MapIcon />
					{result ? "Not on the map" : "No map yet"}
					<ChevronDown className="transition-transform group-data-[state=open]:rotate-180" />
				</Button>
			</CollapsibleTrigger>
			<CollapsibleContent className="basis-full space-y-1 text-sm">
				<p className="text-muted-foreground">
					{!result
						? "This database has no map yet."
						: map.stale === false
							? "This document is not on the map: it has no usable embedding."
							: map.stale
								? "This document is not on the map, and the database changed since the map was computed."
								: "This document is not on the map."}
				</p>
				<Estimate text={estimate.text} />
				<MapControls map={map} ready={estimate.ready} />
			</CollapsibleContent>
		</Collapsible>
	);
}
