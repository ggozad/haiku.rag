"use client";

import { Check, Copy, X } from "lucide-react";
import { useEffect, useState } from "react";
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
import { formatRelativeTime, readableUri } from "@/lib/format";
import { fileName, formatAge } from "@/lib/ingestion";
import type { Job } from "./types";

function duration(job: Job): string {
	if (!job.claimed_at || !job.completed_at) return "—";
	const seconds =
		(Date.parse(job.completed_at) - Date.parse(job.claimed_at)) / 1000;
	return formatAge(seconds);
}

// A job's file name over its full URI, with a button copying the URI. A click
// on the row opens the job, and the button stops its own click from reaching
// the row.
export function FileCell({ job }: { job: Job }) {
	// "Copied", or why the copy failed, until it resets.
	const [copy, setCopy] = useState<string | null>(null);
	useEffect(() => {
		if (copy === null) return;
		const reset = setTimeout(() => setCopy(null), 1500);
		return () => clearTimeout(reset);
	}, [copy]);
	const uri = readableUri(job.uri);
	return (
		<div className="flex min-w-0 items-center gap-1">
			<div className="min-w-0 flex-1">
				<div className="flex min-w-0 items-center gap-1.5 font-medium">
					{job.op === "delete" && <Badge variant="destructive">delete</Badge>}
					<span className="truncate">{fileName(job.uri)}</span>
				</div>
				<div
					className="truncate font-mono text-xs text-muted-foreground select-all"
					title={uri}
				>
					{uri}
				</div>
			</div>
			<Button
				variant="ghost"
				size="icon-xs"
				aria-label="Copy URI"
				title={copy ?? "Copy URI"}
				onClick={(event) => {
					event.stopPropagation();
					if (!navigator.clipboard) {
						setCopy("Copying needs HTTPS or localhost");
						return;
					}
					navigator.clipboard.writeText(job.uri).then(
						() => setCopy("Copied"),
						(error: Error) => setCopy(`Copy failed: ${error.message}`),
					);
				}}
			>
				{copy === "Copied" ? <Check /> : copy ? <X /> : <Copy />}
			</Button>
		</div>
	);
}

export type Column =
	| "file"
	| "source"
	| "attempts"
	| "running"
	| "error"
	| "duration"
	| "finished";

const HEADINGS: Record<Column, string> = {
	file: "File",
	source: "Source",
	attempts: "Attempts",
	running: "Running",
	error: "Error",
	duration: "Took",
	finished: "Finished",
};

function cell(job: Job, column: Column): React.ReactNode {
	switch (column) {
		case "file":
			return <FileCell job={job} />;
		case "source":
			return job.source_id;
		case "attempts":
			return `${job.attempts}/${job.max_attempts}`;
		case "running":
			return job.claimed_at
				? formatAge((Date.now() - Date.parse(job.claimed_at)) / 1000)
				: "—";
		case "error":
			return job.last_error ?? "—";
		case "duration":
			return duration(job);
		case "finished":
			return job.completed_at
				? formatRelativeTime(job.completed_at, true)
				: "—";
	}
}

export default function JobTable({
	jobs,
	columns,
	onOpen,
	action,
	empty,
}: {
	jobs: Job[];
	columns: Column[];
	onOpen: (job: Job) => void;
	action?: (job: Job) => React.ReactNode;
	empty: string;
}) {
	if (jobs.length === 0)
		return <p className="text-sm text-muted-foreground">{empty}</p>;
	return (
		<div className="overflow-x-auto rounded-md border">
			<Table>
				<TableHeader>
					<TableRow>
						{columns.map((column) => (
							<TableHead key={column}>{HEADINGS[column]}</TableHead>
						))}
						{action && <TableHead />}
					</TableRow>
				</TableHeader>
				<TableBody>
					{jobs.map((job) => (
						<TableRow key={job.id}>
							{columns.map((column) => (
								<TableCell
									key={column}
									className={
										column === "file"
											? "max-w-80 cursor-pointer"
											: column === "error"
												? "max-w-80 cursor-pointer truncate"
												: "cursor-pointer"
									}
									title={
										column === "error" ? (job.last_error ?? "") : undefined
									}
									onClick={() => onOpen(job)}
								>
									{cell(job, column)}
								</TableCell>
							))}
							{action && (
								<TableCell className="text-right">{action(job)}</TableCell>
							)}
						</TableRow>
					))}
				</TableBody>
			</Table>
		</div>
	);
}
