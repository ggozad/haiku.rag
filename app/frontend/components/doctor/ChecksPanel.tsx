"use client";

import {
	ChevronRight,
	CircleAlert,
	CircleCheck,
	TriangleAlert,
} from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { type Check, counts, ordered, type Severity } from "@/lib/doctor";
import { useJob } from "./useJob";

const SEVERITY: Record<
	Severity,
	{
		label: string;
		icon: typeof CircleCheck;
		variant: "secondary" | "outline" | "destructive";
	}
> = {
	ok: { label: "ok", icon: CircleCheck, variant: "secondary" },
	warn: { label: "warning", icon: TriangleAlert, variant: "outline" },
	fail: { label: "failed", icon: CircleAlert, variant: "destructive" },
};

function CheckRow({ check }: { check: Check }) {
	const { label, icon: Icon, variant } = SEVERITY[check.severity];
	const body = (
		<div className="flex items-start gap-3 text-sm">
			<Badge variant={variant} className="mt-0.5 shrink-0 gap-1">
				<Icon />
				{label}
			</Badge>
			<div className="min-w-0 space-y-1">
				<div>
					<span className="font-mono text-xs text-muted-foreground">
						{check.name}
					</span>{" "}
					{check.message}
				</div>
				{check.remediation && (
					<div className="text-xs">
						Fix: <code className="font-mono">{check.remediation}</code>
					</div>
				)}
			</div>
		</div>
	);
	if (!check.details.length) {
		return <div className="rounded-md border px-3 py-2">{body}</div>;
	}
	return (
		<Collapsible className="rounded-md border px-3 py-2">
			<CollapsibleTrigger className="group flex w-full items-start gap-1 text-left">
				<ChevronRight className="mt-1 size-3.5 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-90" />
				{body}
			</CollapsibleTrigger>
			<CollapsibleContent>
				<pre className="mt-2 overflow-x-auto whitespace-pre-wrap font-mono text-xs text-muted-foreground">
					{check.details.join("\n")}
				</pre>
			</CollapsibleContent>
		</Collapsible>
	);
}

export default function ChecksPanel({
	source,
	ready,
}: {
	source: string;
	// The memory figure is shown, so the operator can decide.
	ready: boolean;
}) {
	const base = `/api/backend/api/doctor/${encodeURIComponent(source)}`;
	const { job, error, start } = useJob<{ results: Check[] }>(base, "run");
	const [providers, setProviders] = useState(true);
	const busy = job?.status === "queued" || job?.status === "running";
	const checks = job?.result?.results ?? [];
	const tally = counts(checks);

	return (
		<section className="space-y-3">
			<div className="flex flex-wrap items-center gap-3">
				<h2 className="font-semibold">Checks</h2>
				<label
					htmlFor="doctor-providers"
					className="flex items-center gap-2 text-sm"
				>
					<Checkbox
						id="doctor-providers"
						checked={providers}
						onCheckedChange={(value) => setProviders(value === true)}
					/>
					Probe providers and API keys
				</label>
				<Button
					size="sm"
					disabled={busy || !ready}
					onClick={() => start(`${base}/run?providers=${providers}`)}
				>
					{busy ? "Running…" : "Run the doctor"}
				</Button>
				{job?.finished_at && (
					<span className="text-xs text-muted-foreground">
						Finished {new Date(job.finished_at).toLocaleString()}
					</span>
				)}
			</div>
			{error && <p className="text-sm text-destructive">{error}</p>}
			{busy && (
				<ol className="space-y-0.5 text-sm text-muted-foreground">
					{job.status === "queued" && (
						<li>Waiting for another job to finish…</li>
					)}
					{job.progress.map((label, n) => (
						// biome-ignore lint/suspicious/noArrayIndexKey: progress labels can repeat
						<li key={n}>{label}</li>
					))}
				</ol>
			)}
			{job?.status === "failed" && (
				<p className="text-sm text-destructive">
					The doctor failed: {job.error}
				</p>
			)}
			{job?.status === "done" && (
				<>
					<div className="flex gap-2 text-sm">
						<Badge variant="destructive">{tally.fail} failed</Badge>
						<Badge variant="outline">{tally.warn} warnings</Badge>
						<Badge variant="secondary">{tally.ok} ok</Badge>
					</div>
					<div className="space-y-2">
						{ordered(checks).map((check) => (
							<CheckRow key={check.name} check={check} />
						))}
					</div>
				</>
			)}
		</section>
	);
}
