"use client";

import { RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { formatRelativeTime } from "@/lib/format";
import type { Source, Stats } from "./types";

// The configured sources, their queue and dead-letter counts, and Refresh.
export default function SourcesPanel({
	sources,
	stats,
	onRefresh,
}: {
	sources: Source[];
	stats: Stats | undefined;
	onRefresh: (sourceId: string) => void;
}) {
	return (
		<div className="overflow-x-auto rounded-md border">
			<Table>
				<TableHeader>
					<TableRow>
						<TableHead>Source</TableHead>
						<TableHead>Type</TableHead>
						<TableHead>Last polled</TableHead>
						<TableHead>Queued</TableHead>
						<TableHead>Dead</TableHead>
						<TableHead>State</TableHead>
						<TableHead />
					</TableRow>
				</TableHeader>
				<TableBody>
					{sources.map((source) => (
						<TableRow key={source.source_id}>
							<TableCell className="font-medium">{source.source_id}</TableCell>
							<TableCell>{source.type}</TableCell>
							<TableCell>
								{source.last_polled_at
									? formatRelativeTime(source.last_polled_at, true)
									: "never"}
							</TableCell>
							<TableCell className="tabular-nums">
								{stats?.queue_depth_by_source[source.source_id] ?? 0}
							</TableCell>
							<TableCell className="tabular-nums">
								{stats?.dlq_by_source[source.source_id] ?? 0}
							</TableCell>
							<TableCell className="space-x-1">
								{source.circuit_breaker_open && (
									<Badge variant="destructive">breaker open</Badge>
								)}
								{source.last_skip_reason && (
									<Badge variant="outline">
										skipped: {source.last_skip_reason}
									</Badge>
								)}
							</TableCell>
							<TableCell className="text-right">
								<Button
									size="sm"
									variant="outline"
									onClick={() => onRefresh(source.source_id)}
								>
									<RefreshCw />
									Refresh
								</Button>
							</TableCell>
						</TableRow>
					))}
					{sources.length === 0 && (
						<TableRow>
							<TableCell colSpan={7} className="text-muted-foreground">
								No sources are configured.
							</TableCell>
						</TableRow>
					)}
				</TableBody>
			</Table>
		</div>
	);
}
