"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect } from "react";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { doctorHref } from "@/lib/addresses";
import { useDatabases } from "@/lib/databases";
import { formatBytes, scanBytes } from "@/lib/doctor";
import { useJson } from "@/lib/useJson";
import ChecksPanel from "./ChecksPanel";
import CorpusMap from "./CorpusMap";

export default function DoctorPage() {
	const { info } = useDatabases();
	const databases = info?.databases ?? [];
	const router = useRouter();
	const named = useParams<{ db?: string[] }>().db?.[0] ?? null;
	const first = databases[0]?.name;
	useEffect(() => {
		if (named === null && first) router.replace(doctorHref(first));
	}, [named, first, router]);
	const known =
		info && named !== null && databases.some((db) => db.name === named);
	const source = known ? named : null;

	const details = useJson<{
		info: {
			embeddings: { vector_dim: number | null };
			tables: { name: string; num_rows: number }[];
		};
	}>(
		source
			? `/api/backend/api/inspect/${encodeURIComponent(source)}/info`
			: null,
	);
	const dimension = details.data?.info.embeddings.vector_dim;
	const chunks = details.data?.info.tables.find(
		(table) => table.name === "chunks",
	)?.num_rows;
	// The operator sees the memory figure before starting, or why it is unknown.
	const estimated = details.data !== null || details.error !== null;

	return (
		<div className="h-full overflow-y-auto">
			<div className="mx-auto max-w-5xl space-y-6 px-4 py-6">
				<header className="flex flex-wrap items-center gap-3">
					<h1 className="text-lg font-semibold">Doctor</h1>
					<Select
						value={source ?? ""}
						onValueChange={(name) => router.push(doctorHref(name))}
					>
						<SelectTrigger className="w-56" aria-label="Database">
							<SelectValue placeholder="Database" />
						</SelectTrigger>
						<SelectContent>
							{databases.map((db) => (
								<SelectItem key={db.name} value={db.name}>
									{db.name}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
					{chunks !== undefined && dimension ? (
						<span className="text-sm text-muted-foreground">
							Reading the vectors holds at least{" "}
							{formatBytes(scanBytes(chunks, dimension))} in memory.
						</span>
					) : details.error ? (
						<span className="text-sm text-muted-foreground">
							The memory needed is unknown: {details.error}
						</span>
					) : null}
				</header>
				{info && named !== null && !known && (
					<p className="text-sm text-muted-foreground">
						No database is named {named}.
					</p>
				)}
				<p className="text-xs text-muted-foreground">
					Results are kept until the next run or a restart of the backend, and
					one job runs at a time across databases.
				</p>
				{source && (
					<CorpusMap key={`map-${source}`} source={source} ready={estimated} />
				)}
				{source && (
					<ChecksPanel key={source} source={source} ready={estimated} />
				)}
			</div>
		</div>
	);
}
