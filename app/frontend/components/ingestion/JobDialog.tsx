"use client";

import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import type { Job } from "./types";

export default function JobDialog({
	job,
	onClose,
}: {
	job: Job | null;
	onClose: () => void;
}) {
	return (
		<Dialog open={job !== null} onOpenChange={(open) => !open && onClose()}>
			<DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-2xl">
				<DialogHeader>
					<DialogTitle className="break-all">{job?.uri}</DialogTitle>
					<DialogDescription>Job {job?.id}</DialogDescription>
				</DialogHeader>
				{job && (
					<dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
						{Object.entries(job).map(([key, value]) => (
							<div key={key} className="contents">
								<dt className="text-muted-foreground">{key}</dt>
								<dd className="whitespace-pre-wrap break-all font-mono text-xs">
									{value === null
										? "—"
										: typeof value === "object"
											? JSON.stringify(value, null, 2)
											: String(value)}
								</dd>
							</div>
						))}
					</dl>
				)}
			</DialogContent>
		</Dialog>
	);
}
