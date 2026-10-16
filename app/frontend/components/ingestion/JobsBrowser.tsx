"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { formatRelativeTime } from "@/lib/format";
import { ingester } from "@/lib/ingestion";
import JobDialog from "./JobDialog";
import type { Job } from "./types";

const PAGE = 50;
const STATUSES = ["any", "queued", "claimed", "succeeded", "dead"];

// Every job, filtered and paged. Read on demand, not polled.
export default function JobsBrowser({
	sources,
	revision,
}: {
	sources: string[];
	// Changes after an operator action, so the page reads again.
	revision: number;
}) {
	const [status, setStatus] = useState("any");
	const [source, setSource] = useState("any");
	const [uri, setUri] = useState("");
	const [offset, setOffset] = useState(0);
	const [jobs, setJobs] = useState<Job[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [open, setOpen] = useState<Job | null>(null);
	// Only the latest read lands: a slow one never replaces a newer page.
	const latest = useRef(0);

	const load = useCallback(
		(signal?: AbortSignal) => {
			const query = new URLSearchParams({
				limit: String(PAGE),
				offset: String(offset),
			});
			if (status !== "any") query.set("status", status);
			if (source !== "any") query.set("source_id", source);
			if (uri.trim()) query.set("uri", uri.trim());
			const mine = ++latest.current;
			ingester(`jobs?${query}`, signal ? { signal } : {})
				.then((data) => {
					if (mine !== latest.current) return;
					setJobs(data as Job[]);
					setError(null);
				})
				.catch((err: Error) => {
					if (mine === latest.current) setError(err.message);
				});
		},
		[status, source, uri, offset],
	);

	// biome-ignore lint/correctness/useExhaustiveDependencies: revision asks for a reread
	useEffect(() => {
		const controller = new AbortController();
		load(controller.signal);
		return () => controller.abort();
	}, [load, revision]);

	return (
		<section className="space-y-3">
			<h2 className="text-sm font-semibold">All jobs</h2>
			<div className="flex flex-wrap items-center gap-2">
				<Select
					value={status}
					onValueChange={(value) => {
						setStatus(value);
						setOffset(0);
					}}
				>
					<SelectTrigger className="w-36" aria-label="Status">
						<SelectValue />
					</SelectTrigger>
					<SelectContent>
						{STATUSES.map((s) => (
							<SelectItem key={s} value={s}>
								{s === "any" ? "Any status" : s}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
				<Select
					value={source}
					onValueChange={(value) => {
						setSource(value);
						setOffset(0);
					}}
				>
					<SelectTrigger className="w-44" aria-label="Source">
						<SelectValue />
					</SelectTrigger>
					<SelectContent>
						<SelectItem value="any">Any source</SelectItem>
						{sources.map((s) => (
							<SelectItem key={s} value={s}>
								{s}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
				<Input
					id="jobs-uri"
					className="w-72"
					placeholder="Exact URI"
					value={uri}
					onChange={(event) => {
						setUri(event.target.value);
						setOffset(0);
					}}
				/>
				<Button variant="outline" size="sm" onClick={() => load()}>
					Reload
				</Button>
			</div>
			{error && <p className="text-sm text-destructive">{error}</p>}
			<div className="overflow-x-auto rounded-md border">
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead>Status</TableHead>
							<TableHead>Op</TableHead>
							<TableHead>URI</TableHead>
							<TableHead>Source</TableHead>
							<TableHead>Attempts</TableHead>
							<TableHead>Enqueued</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{jobs.map((job) => (
							<TableRow
								key={job.id}
								className="cursor-pointer"
								onClick={() => setOpen(job)}
							>
								<TableCell>{job.status}</TableCell>
								<TableCell>{job.op}</TableCell>
								<TableCell className="max-w-96 truncate">{job.uri}</TableCell>
								<TableCell>{job.source_id}</TableCell>
								<TableCell className="tabular-nums">
									{job.attempts}/{job.max_attempts}
								</TableCell>
								<TableCell className="text-muted-foreground">
									{formatRelativeTime(job.enqueued_at, true)}
								</TableCell>
							</TableRow>
						))}
						{jobs.length === 0 && (
							<TableRow>
								<TableCell colSpan={6} className="text-muted-foreground">
									No jobs match.
								</TableCell>
							</TableRow>
						)}
					</TableBody>
				</Table>
			</div>
			<div className="flex items-center gap-2 text-sm">
				<Button
					variant="outline"
					size="sm"
					disabled={offset === 0}
					onClick={() => setOffset(Math.max(0, offset - PAGE))}
				>
					Previous
				</Button>
				<span className="text-muted-foreground tabular-nums">
					{offset + 1}–{offset + jobs.length}
				</span>
				<Button
					variant="outline"
					size="sm"
					disabled={jobs.length < PAGE}
					onClick={() => setOffset(offset + PAGE)}
				>
					Next
				</Button>
			</div>
			<JobDialog job={open} onClose={() => setOpen(null)} />
		</section>
	);
}
