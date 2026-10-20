"use client";

import { ChevronRight, RotateCcw, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { formatRelativeTime } from "@/lib/format";
import { ingester, Poller, type Snapshot } from "@/lib/ingestion";
import JobDialog from "./JobDialog";
import JobsBrowser from "./JobsBrowser";
import JobTable from "./JobTable";
import QueueSummary from "./QueueSummary";
import SourcesPanel from "./SourcesPanel";
import type { Health, Job, Providers, Source, Stats } from "./types";

function Section({
	title,
	error,
	children,
}: {
	title: string;
	error?: string;
	children: React.ReactNode;
}) {
	return (
		<section className="min-w-0 space-y-2">
			<h2 className="text-sm font-semibold">{title}</h2>
			{error ? <p className="text-sm text-destructive">{error}</p> : children}
		</section>
	);
}

// A refresh answers once its source's discovery sweep ends.
const REFRESH_TIMEOUT_MS = 300_000;

function Lazy({
	title,
	path,
	render,
	onLoad,
}: {
	title: string;
	path: string;
	render: (data: unknown) => React.ReactNode;
	onLoad?: (data: unknown) => void;
}) {
	const [data, setData] = useState<unknown>(null);
	const [error, setError] = useState<string | null>(null);
	return (
		<Collapsible
			className="rounded-md border"
			onOpenChange={(open) => {
				if (!open) return;
				ingester(path)
					.then((value) => {
						setData(value);
						setError(null);
						onLoad?.(value);
					})
					.catch((err: Error) => setError(err.message));
			}}
		>
			<CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-semibold">
				<ChevronRight className="size-3.5 transition-transform group-data-[state=open]:rotate-90" />
				{title}
			</CollapsibleTrigger>
			<CollapsibleContent className="border-t p-3">
				{error ? (
					<p className="text-sm text-destructive">{error}</p>
				) : data === null ? (
					<p className="text-sm text-muted-foreground">Loading…</p>
				) : (
					render(data)
				)}
			</CollapsibleContent>
		</Collapsible>
	);
}

export default function IngestionPage() {
	const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
	const [notice, setNotice] = useState<string | null>(null);
	const [database, setDatabase] = useState<string | null>(null);
	const [confirming, setConfirming] = useState<Job | null>(null);
	const [viewing, setViewing] = useState<Job | null>(null);
	const [revision, setRevision] = useState(0);
	const poller = useMemo(() => new Poller(setSnapshot), []);

	useEffect(() => {
		const visible = () => {
			if (document.hidden) poller.stop();
			else poller.start();
		};
		visible();
		document.addEventListener("visibilitychange", visible);
		// A failure leaves the header to the database panel.
		ingester("database")
			.then((info) => setDatabase((info as { path: string }).path))
			.catch(() => {});
		return () => {
			document.removeEventListener("visibilitychange", visible);
			poller.stop();
		};
	}, [poller]);

	const act = useCallback(
		async (action: () => Promise<string | null>) => {
			poller.interrupt();
			try {
				setNotice(await action());
			} catch (error) {
				setNotice(error instanceof Error ? error.message : String(error));
			}
			setRevision((n) => n + 1);
			await poller.refresh();
		},
		[poller],
	);

	const health = snapshot?.health.data as Health | undefined;
	const stats = snapshot?.stats.data as Stats | undefined;
	const sources = (snapshot?.sources.data as Source[] | undefined) ?? [];
	const providers = snapshot?.providers.data as Providers | undefined;
	const active = (snapshot?.active.data as Job[] | undefined) ?? [];
	const dead = (snapshot?.dead.data as Job[] | undefined) ?? [];
	const succeeded = (snapshot?.succeeded.data as Job[] | undefined) ?? [];
	const unreachable = snapshot?.health.error;

	return (
		<div className="h-full overflow-y-auto">
			<div className="mx-auto max-w-6xl space-y-6 px-4 py-6">
				<header className="flex flex-wrap items-center gap-3">
					<h1 className="text-lg font-semibold">Ingestion</h1>
					{unreachable ? (
						<Badge variant="destructive" title={unreachable}>
							Unreachable
						</Badge>
					) : health ? (
						<Badge
							variant={health.status === "ok" ? "secondary" : "destructive"}
						>
							{health.status}
						</Badge>
					) : null}
					{health && (
						<span className="text-sm text-muted-foreground">
							{health.workers_alive}/{health.worker_count} workers ·{" "}
							{health.pollers_alive}/{health.poller_count} pollers
						</span>
					)}
					{health?.worker_breaker_open && (
						<Badge variant="destructive">
							Worker breaker open after{" "}
							{health.worker_breaker_consecutive_failures} failures
						</Badge>
					)}
					{database && (
						<span className="text-sm text-muted-foreground">
							Writes {database}
						</span>
					)}
					<span className="ml-auto text-xs text-muted-foreground">
						{!snapshot
							? "Loading…"
							: snapshot.healthyAt === null
								? "Not reached yet"
								: `Updated ${formatRelativeTime(new Date(snapshot.healthyAt).toISOString())}`}
					</span>
				</header>
				{unreachable && (
					<Alert variant="destructive">
						<AlertDescription>{unreachable}</AlertDescription>
					</Alert>
				)}
				{notice && (
					<Alert>
						<AlertDescription>{notice}</AlertDescription>
					</Alert>
				)}

				<Section title="Queue" error={snapshot?.stats.error}>
					{stats && health && <QueueSummary health={health} stats={stats} />}
				</Section>

				<Section title="Sources" error={snapshot?.sources.error}>
					<SourcesPanel
						sources={sources}
						stats={stats}
						onRefresh={(source) =>
							act(async () => {
								const result = (await ingester(
									`sources/${encodeURIComponent(source)}/refresh`,
									{ method: "POST" },
									REFRESH_TIMEOUT_MS,
								)) as { refreshed: boolean };
								return result.refreshed
									? `Refreshed ${source}.`
									: `Refresh of ${source} skipped or failed.`;
							})
						}
					/>
				</Section>

				<Section title="Providers" error={snapshot?.providers.error}>
					<ul className="space-y-1 text-sm">
						{providers?.docling_serve.map((endpoint) => (
							<li key={endpoint.base_url} className="flex items-center gap-2">
								<Badge
									variant={endpoint.reachable ? "secondary" : "destructive"}
								>
									{endpoint.reachable ? "reachable" : "unreachable"}
								</Badge>
								<span className="font-mono text-xs">{endpoint.base_url}</span>
								{endpoint.status_code !== null && (
									<span className="text-muted-foreground">
										HTTP {endpoint.status_code}
									</span>
								)}
								{endpoint.error && (
									<span className="text-muted-foreground">
										{endpoint.error}
									</span>
								)}
							</li>
						))}
						{providers && providers.docling_serve.length === 0 && (
							<li className="text-muted-foreground">
								No docling-serve endpoints are configured.
							</li>
						)}
					</ul>
				</Section>

				<Section title="Active jobs" error={snapshot?.active.error}>
					<JobTable
						jobs={active}
						columns={["op", "uri", "source", "attempts", "worker", "claimed"]}
						onOpen={setViewing}
						action={(job) => (
							<Button
								size="sm"
								variant="outline"
								onClick={() => setConfirming(job)}
							>
								<X />
								Cancel
							</Button>
						)}
						empty="Nothing is being ingested."
					/>
				</Section>

				<Section title="Dead letter" error={snapshot?.dead.error}>
					<JobTable
						jobs={dead}
						columns={["op", "uri", "source", "attempts", "error"]}
						onOpen={setViewing}
						action={(job) => (
							<Button
								size="sm"
								variant="outline"
								onClick={() =>
									act(async () => {
										await ingester(`jobs/${encodeURIComponent(job.id)}/retry`, {
											method: "POST",
										});
										return `Queued ${job.uri} again.`;
									})
								}
							>
								<RotateCcw />
								Retry
							</Button>
						)}
						empty="No job has failed for good."
					/>
				</Section>

				<Section
					title="Latest succeeded, by enqueue time"
					error={snapshot?.succeeded.error}
				>
					<JobTable
						jobs={succeeded}
						columns={["op", "uri", "source", "duration", "finished"]}
						onOpen={setViewing}
						empty="No job has succeeded yet."
					/>
				</Section>

				<JobsBrowser
					sources={sources.map((s) => s.source_id)}
					revision={revision}
				/>

				<div className="space-y-2">
					<Lazy
						title="Database"
						path="database"
						onLoad={(info) => setDatabase((info as { path: string }).path)}
						render={(data) => (
							<pre className="overflow-x-auto font-mono text-xs">
								{JSON.stringify(data, null, 2)}
							</pre>
						)}
					/>
					<Lazy
						title="Configuration"
						path="config"
						render={(data) => (
							<pre className="overflow-x-auto font-mono text-xs">
								{(data as { yaml: string }).yaml}
							</pre>
						)}
					/>
				</div>
			</div>

			<Dialog
				open={confirming !== null}
				onOpenChange={(open) => !open && setConfirming(null)}
			>
				<DialogContent className="sm:max-w-md">
					<DialogHeader>
						<DialogTitle>Cancel this job?</DialogTitle>
						<DialogDescription className="break-all">
							{confirming?.uri} leaves the queue. A conversion already running
							finishes and may still write the document.
						</DialogDescription>
					</DialogHeader>
					<DialogFooter>
						<Button variant="outline" onClick={() => setConfirming(null)}>
							Keep it
						</Button>
						<Button
							variant="destructive"
							onClick={() => {
								const job = confirming;
								setConfirming(null);
								if (job) {
									act(async () => {
										await ingester(`jobs/${encodeURIComponent(job.id)}`, {
											method: "DELETE",
										});
										return `Cancelled ${job.uri}.`;
									});
								}
							}}
						>
							Cancel the job
						</Button>
					</DialogFooter>
				</DialogContent>
			</Dialog>
			<JobDialog job={viewing} onClose={() => setViewing(null)} />
		</div>
	);
}
