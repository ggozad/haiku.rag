"use client";

import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { useJson } from "@/lib/useJson";

interface Info {
	info: {
		exists: boolean;
		stored_version: string;
		embeddings: { provider: string; name: string; vector_dim: number | null };
		tables: {
			name: string;
			exists: boolean;
			num_rows: number;
			total_bytes: number;
			num_versions: number;
		}[];
		vector_index: {
			exists: boolean;
			indexed_rows: number;
			unindexed_rows: number;
		};
		pending_migrations: { version: string; description: string }[];
		packages: Record<string, string>;
	};
	tags: Record<
		string,
		{
			complete: boolean;
			tables: Record<string, number>;
			missing_tables: string[];
		}
	> | null;
	history?: Record<
		string,
		{ version: number; timestamp: string; tags: string[] }[]
	>;
	unavailable?: string;
}

function bytes(n: number): string {
	if (n < 1024) return `${n} B`;
	const units = ["KB", "MB", "GB", "TB"];
	let value = n / 1024;
	let unit = 0;
	while (value >= 1024 && unit < units.length - 1) {
		value /= 1024;
		unit++;
	}
	return `${value.toFixed(1)} ${units[unit]}`;
}

export default function InfoDialog({
	source,
	open,
	onOpenChange,
}: {
	source: string;
	open: boolean;
	onOpenChange: (open: boolean) => void;
}) {
	const { data, error } = useJson<Info>(
		open ? `/api/backend/api/inspect/${encodeURIComponent(source)}/info` : null,
	);
	const info = data?.info;
	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-3xl">
				<DialogHeader>
					<DialogTitle>{source}</DialogTitle>
					<DialogDescription>
						{info
							? `haiku.rag ${info.stored_version} · ${info.embeddings.provider}/${info.embeddings.name}, ${info.embeddings.vector_dim} dimensions`
							: "Database information"}
					</DialogDescription>
				</DialogHeader>
				{error && <p className="text-sm text-destructive">{error}</p>}
				{info && !info.exists && (
					<p className="text-sm text-destructive">
						The database does not exist.
					</p>
				)}
				{info?.exists && (
					<div className="space-y-4 text-sm">
						<table className="w-full text-left">
							<thead className="text-xs text-muted-foreground">
								<tr>
									<th className="font-normal">Table</th>
									<th className="font-normal">Rows</th>
									<th className="font-normal">Size</th>
									<th className="font-normal">Versions</th>
								</tr>
							</thead>
							<tbody className="tabular-nums">
								{info.tables.map((table) => (
									<tr key={table.name} className="border-t">
										<td>{table.name}</td>
										<td>{table.exists ? table.num_rows : "missing"}</td>
										<td>{table.exists ? bytes(table.total_bytes) : ""}</td>
										<td>{table.exists ? table.num_versions : ""}</td>
									</tr>
								))}
							</tbody>
						</table>
						<p>
							Vector index:{" "}
							{info.vector_index.exists
								? `${info.vector_index.indexed_rows} rows indexed, ${info.vector_index.unindexed_rows} not yet`
								: "none"}
						</p>
						{info.pending_migrations.length > 0 && (
							<div>
								<p className="font-medium">Pending migrations</p>
								<ul className="list-disc pl-5">
									{info.pending_migrations.map((m) => (
										<li key={m.version}>
											{m.version}: {m.description}
										</li>
									))}
								</ul>
							</div>
						)}
						{data?.unavailable && (
							<p className="text-muted-foreground">
								Tags and history are unavailable: {data.unavailable}
							</p>
						)}
						{data?.tags && (
							<div>
								<p className="font-medium">Tags</p>
								{Object.keys(data.tags).length === 0 ? (
									<p className="text-muted-foreground">No tags.</p>
								) : (
									<ul>
										{Object.entries(data.tags).map(([name, tag]) => (
											<li key={name}>
												{name}
												{!tag.complete &&
													` (missing on ${tag.missing_tables.join(", ")})`}
											</li>
										))}
									</ul>
								)}
							</div>
						)}
						{data?.history &&
							Object.entries(data.history).map(([table, versions]) => (
								<details key={table}>
									<summary className="cursor-pointer">{table} history</summary>
									<ul className="mt-1 font-mono text-xs">
										{versions.map((v) => (
											<li key={v.version}>
												v{v.version}: {v.timestamp}
												{v.tags.length > 0 && ` ← ${v.tags.join(", ")}`}
											</li>
										))}
									</ul>
								</details>
							))}
						<p className="text-xs text-muted-foreground">
							{Object.entries(info.packages)
								.map(([name, version]) => `${name} ${version}`)
								.join(" · ")}
						</p>
					</div>
				)}
			</DialogContent>
		</Dialog>
	);
}
