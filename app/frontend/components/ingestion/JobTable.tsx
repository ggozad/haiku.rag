"use client";

import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { formatRelativeTime } from "@/lib/format";
import { formatAge } from "@/lib/ingestion";
import type { Job } from "./types";

function duration(job: Job): string {
	if (!job.claimed_at || !job.completed_at) return "—";
	const seconds =
		(Date.parse(job.completed_at) - Date.parse(job.claimed_at)) / 1000;
	return formatAge(seconds);
}

export type Column =
	| "op"
	| "uri"
	| "source"
	| "attempts"
	| "worker"
	| "claimed"
	| "error"
	| "duration"
	| "finished";

const HEADINGS: Record<Column, string> = {
	op: "Op",
	uri: "URI",
	source: "Source",
	attempts: "Attempts",
	worker: "Worker",
	claimed: "Claimed",
	error: "Error",
	duration: "Took",
	finished: "Finished",
};

function cell(job: Job, column: Column): React.ReactNode {
	switch (column) {
		case "op":
			return job.op;
		case "uri":
			return job.uri;
		case "source":
			return job.source_id;
		case "attempts":
			return `${job.attempts}/${job.max_attempts}`;
		case "worker":
			return job.claimed_by ?? "—";
		case "claimed":
			return job.claimed_at ? formatRelativeTime(job.claimed_at, true) : "—";
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
										column === "uri" || column === "error"
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
					{jobs.length === 0 && (
						<TableRow>
							<TableCell
								colSpan={columns.length + (action ? 1 : 0)}
								className="text-muted-foreground"
							>
								{empty}
							</TableCell>
						</TableRow>
					)}
				</TableBody>
			</Table>
		</div>
	);
}
