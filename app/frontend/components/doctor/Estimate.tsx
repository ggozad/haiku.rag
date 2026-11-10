"use client";

import { scanBytes } from "@/lib/doctor";
import { formatBytes } from "@/lib/format";
import { withBasePath } from "@/lib/paths";
import { useJson } from "@/lib/useJson";

// The memory needed to read every chunk vector. `ready` once the figure, or
// why it is unknown, is shown, so the operator can decide before starting a
// job.
export function useEstimate(source: string | null) {
	const details = useJson<{
		info: {
			embeddings: { vector_dim: number | null };
			tables: { name: string; num_rows: number }[];
		};
	}>(
		source === null
			? null
			: withBasePath(
					`/api/backend/api/inspect/${encodeURIComponent(source)}/info`,
				),
	);
	const dimension = details.data?.info.embeddings.vector_dim;
	const chunks = details.data?.info.tables.find(
		(table) => table.name === "chunks",
	)?.num_rows;
	const text =
		chunks !== undefined && dimension
			? `Reading the vectors holds at least ${formatBytes(scanBytes(chunks, dimension))} in memory.`
			: details.error
				? `The memory needed is unknown: ${details.error}`
				: null;
	return { ready: details.data !== null || details.error !== null, text };
}

export function Estimate({ text }: { text: string | null }) {
	return text ? (
		<span className="text-sm text-muted-foreground">{text}</span>
	) : null;
}
