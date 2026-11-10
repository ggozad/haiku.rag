"use client";

import DatabaseTables, {
	type DatabaseInfo,
	describeDatabase,
} from "@/components/DatabaseInfo";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { withBasePath } from "@/lib/paths";
import { useJson } from "@/lib/useJson";

interface Info {
	info: DatabaseInfo;
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
		open
			? withBasePath(
					`/api/backend/api/inspect/${encodeURIComponent(source)}/info`,
				)
			: null,
	);
	const info = data?.info;
	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-3xl">
				<DialogHeader>
					<DialogTitle>{source}</DialogTitle>
					<DialogDescription>
						{info ? describeDatabase(info) : "Database information"}
					</DialogDescription>
				</DialogHeader>
				{error && <p className="text-sm text-destructive">{error}</p>}
				{info && <DatabaseTables info={info} />}
				{info?.exists && (
					<div className="mt-4 space-y-4 text-sm">
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
												v{v.version}: {new Date(v.timestamp).toLocaleString()}
												{v.tags.length > 0 && ` ← ${v.tags.join(", ")}`}
											</li>
										))}
									</ul>
								</details>
							))}
					</div>
				)}
			</DialogContent>
		</Dialog>
	);
}
