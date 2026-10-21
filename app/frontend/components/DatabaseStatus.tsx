"use client";

import { Badge } from "@/components/ui/badge";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { useDatabases } from "@/lib/databases";

export default function DatabaseStatus() {
	const { info, error } = useDatabases();
	if (error) {
		return (
			<Badge variant="destructive" title={error}>
				Backend unavailable
			</Badge>
		);
	}
	if (!info) return null;

	return (
		<div className="flex items-center gap-1.5">
			{info.databases.map((db) => (
				<Tooltip key={db.name}>
					<TooltipTrigger asChild>
						<Badge variant={db.available ? "secondary" : "destructive"}>
							{db.name}
						</Badge>
					</TooltipTrigger>
					<TooltipContent>
						{db.available
							? `${db.documents} documents, ${db.chunks} chunks`
							: db.error}
					</TooltipContent>
				</Tooltip>
			))}
		</div>
	);
}
