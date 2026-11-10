import { formatBytes } from "@/lib/format";

export interface DatabaseInfo {
	path: string;
	exists: boolean;
	stored_version: string;
	embeddings: { provider: string; name: string; vector_dim: number | null };
	tables: {
		name: string;
		exists: boolean;
		num_rows: number;
		total_bytes: number;
		num_versions: number;
		latest_version_at: string | null;
	}[];
	vector_index: {
		exists: boolean;
		indexed_rows: number;
		unindexed_rows: number;
	};
	pending_migrations: { version: string; description: string }[];
	packages: Record<string, string>;
}

export function describeDatabase(info: DatabaseInfo): string {
	return `haiku.rag ${info.stored_version} · ${info.embeddings.provider}/${info.embeddings.name}, ${info.embeddings.vector_dim} dimensions`;
}

export default function DatabaseTables({ info }: { info: DatabaseInfo }) {
	if (!info.exists) {
		return (
			<p className="text-sm text-destructive">The database does not exist.</p>
		);
	}
	return (
		<div className="space-y-4 text-sm">
			<table className="w-full text-left">
				<thead className="text-xs text-muted-foreground">
					<tr>
						<th className="font-normal">Table</th>
						<th className="font-normal">Rows</th>
						<th className="font-normal">Size</th>
						<th className="font-normal">Versions</th>
						<th className="font-normal">Last written</th>
					</tr>
				</thead>
				<tbody className="tabular-nums">
					{info.tables.map((table) => (
						<tr key={table.name} className="border-t">
							<td>{table.name}</td>
							<td>{table.exists ? table.num_rows : "missing"}</td>
							<td>{table.exists ? formatBytes(table.total_bytes) : ""}</td>
							<td>{table.exists ? table.num_versions : ""}</td>
							<td>
								{table.latest_version_at
									? new Date(table.latest_version_at).toLocaleString()
									: ""}
							</td>
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
			<p className="text-xs text-muted-foreground">
				{Object.entries(info.packages)
					.map(([name, version]) => `${name} ${version}`)
					.join(" · ")}
			</p>
		</div>
	);
}
