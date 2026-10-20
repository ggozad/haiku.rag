"use client";

import { formatAge } from "@/lib/ingestion";
import type { Health, Stats } from "./types";

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
	return (
		<div className="rounded-md border px-3 py-2">
			<div className="text-xs text-muted-foreground">{label}</div>
			<div className="text-lg font-semibold tabular-nums">{value}</div>
		</div>
	);
}

// The queue's counts, throughput, busy workers and oldest queued job.
export default function QueueSummary({
	health,
	stats,
}: {
	health: Health;
	stats: Stats;
}) {
	return (
		<div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
			{["queued", "claimed", "succeeded", "dead"].map((status) => (
				<Stat
					key={status}
					label={status}
					value={health.queue_counts[status] ?? 0}
				/>
			))}
			<Stat
				label="done in 5m / 1h"
				value={`${stats.throughput.succeeded_5m} / ${stats.throughput.succeeded_1h}`}
			/>
			<Stat
				label="busy workers"
				value={`${stats.workers.busy}/${stats.workers.total}`}
			/>
			<Stat
				label="oldest queued"
				value={formatAge(stats.oldest_queued_age_s)}
			/>
		</div>
	);
}
