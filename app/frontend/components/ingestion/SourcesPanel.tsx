"use client";

import { Loader2, RefreshCw, RotateCcw } from "lucide-react";
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
import {
	completion,
	FILE_STATES,
	ingestedDatabases,
	unsupportedFiles,
} from "@/lib/ingestion";
import type { Source, Stats } from "./types";

// The configured sources, how much of each is ingested, their queued, dead and
// rejected counts, Scan now and Retry failed.
// The database column shows when the ingester writes more than one.
export default function SourcesPanel({
	sources,
	stats,
	busy,
	onScan,
	onRetry,
}: {
	sources: Source[];
	stats: Stats | undefined;
	busy: string | null;
	onScan: (sourceId: string) => void;
	onRetry: (sourceId: string) => void;
}) {
	const showDatabase = ingestedDatabases(sources).length > 1;
	return (
		<div className="overflow-x-auto rounded-md border">
			<Table>
				<TableHeader>
					<TableRow>
						<TableHead>Source</TableHead>
						{showDatabase && <TableHead>Database</TableHead>}
						<TableHead>Type</TableHead>
						<TableHead>Last polled</TableHead>
						<TableHead>Files</TableHead>
						<TableHead>Queued</TableHead>
						<TableHead>Dead</TableHead>
						<TableHead>Rejected</TableHead>
						<TableHead>State</TableHead>
						<TableHead />
					</TableRow>
				</TableHeader>
				<TableBody>
					{sources.map((source) => {
						const dead = stats?.dlq_by_source[source.source_id] ?? 0;
						const working = busy === source.source_id;
						const healthy =
							!source.circuit_breaker_open &&
							!source.last_skip_reason &&
							!source.last_error;
						return (
							<TableRow key={source.source_id}>
								<TableCell className="font-medium">
									{source.source_id}
								</TableCell>
								{showDatabase && (
									<TableCell>{source.database ?? "—"}</TableCell>
								)}
								<TableCell>{source.type}</TableCell>
								<TableCell>
									{source.last_polled_at
										? formatRelativeTime(source.last_polled_at, true)
										: "never"}
								</TableCell>
								<TableCell>
									<Completion source={source} />
								</TableCell>
								<TableCell className="tabular-nums">
									{stats?.queue_depth_by_source[source.source_id] ?? 0}
								</TableCell>
								<TableCell className="tabular-nums">{dead}</TableCell>
								<TableCell className="tabular-nums">
									{stats?.rejected_by_source[source.source_id] ?? 0}
								</TableCell>
								<TableCell className="space-x-1">
									{healthy && <span className="text-muted-foreground">ok</span>}
									{source.circuit_breaker_open && (
										<Badge variant="destructive">breaker open</Badge>
									)}
									{source.last_skip_reason && (
										<Badge variant="outline">
											skipped: {source.last_skip_reason}
										</Badge>
									)}
									{source.last_error && (
										<>
											<Badge
												variant="destructive"
												title={
													source.last_error_at
														? formatRelativeTime(source.last_error_at, true)
														: undefined
												}
											>
												discovery failed
											</Badge>
											<span className="text-xs text-destructive">
												{source.last_error}
											</span>
										</>
									)}
									<SkippedFiles counts={source.unsupported_files} />
								</TableCell>
								<TableCell className="space-x-1 text-right whitespace-nowrap">
									<Button
										size="sm"
										variant="outline"
										disabled={busy !== null}
										title="Run a discovery sweep of this source"
										onClick={() => onScan(source.source_id)}
									>
										{working ? (
											<Loader2 className="animate-spin" />
										) : (
											<RefreshCw />
										)}
										{working ? "Scanning…" : "Scan now"}
									</Button>
									<Button
										size="sm"
										variant="outline"
										disabled={busy !== null || dead === 0}
										onClick={() => onRetry(source.source_id)}
									>
										<RotateCcw />
										Retry failed
									</Button>
								</TableCell>
							</TableRow>
						);
					})}
					{sources.length === 0 && (
						<TableRow>
							<TableCell
								colSpan={showDatabase ? 10 : 9}
								className="text-muted-foreground"
							>
								No sources are configured.
							</TableCell>
						</TableRow>
					)}
				</TableBody>
			</Table>
		</div>
	);
}

const FILE_COLORS = {
	ingested: "bg-emerald-500",
	pending: "bg-sky-500",
	failed: "bg-destructive",
	waiting: "bg-muted-foreground/30",
};

function Completion({ source }: { source: Source }) {
	const percent = completion(source);
	if (percent === null) return <span className="text-muted-foreground">—</span>;
	return (
		<div className="min-w-48 space-y-1">
			<div className="flex items-center gap-2">
				<div
					role="img"
					aria-label={`${percent}% of ${source.known} files ingested`}
					className="flex h-2 flex-1 overflow-hidden rounded-full bg-muted"
				>
					{FILE_STATES.map((state) => (
						<span
							key={state}
							title={`${source[state]} ${state}`}
							className={FILE_COLORS[state]}
							style={{ width: `${(100 * source[state]) / source.known}%` }}
						/>
					))}
				</div>
				<span className="text-xs tabular-nums">{percent}%</span>
			</div>
			<div className="flex flex-wrap gap-x-2 text-xs text-muted-foreground tabular-nums">
				{FILE_STATES.map((state) => (
					<span key={state} className="inline-flex items-center gap-1">
						<span className={`size-2 rounded-full ${FILE_COLORS[state]}`} />
						{source[state]} {state}
					</span>
				))}
			</div>
		</div>
	);
}

function SkippedFiles({ counts }: { counts: Record<string, number> }) {
	const skipped = unsupportedFiles(counts);
	if (!skipped) return null;
	return (
		<Badge variant="outline" title={skipped.detail}>
			{skipped.label}
		</Badge>
	);
}
