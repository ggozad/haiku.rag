"use client";

import Link from "next/link";
import {
	Estimate,
	MapControls,
	useEstimate,
	useMapJob,
} from "@/components/doctor/MapJob";
import { Button } from "@/components/ui/button";

// A document's place on its database's map, or why it has none.
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
				<Link href={href}>Show on map</Link>
			</Button>
		);
	return (
		<div className="space-y-1 text-sm">
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
		</div>
	);
}
