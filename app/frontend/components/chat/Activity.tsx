"use client";

import { ChevronRight, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { activitySummary, type Step } from "@/lib/chat";
import ToolCallCard from "./ToolCallCard";

// One stretch of reasoning and tool calls: open while it is the live end of a
// running answer, folded to a summary line once it is done.
export default function Activity({
	steps,
	results,
	live,
	running,
	sources,
	onSources,
}: {
	steps: Step[];
	results: Map<string, string>;
	live: boolean;
	running: boolean;
	sources: number;
	onSources: () => void;
}) {
	const [open, setOpen] = useState(live);
	useEffect(() => setOpen(live), [live]);

	return (
		<Collapsible open={open} onOpenChange={setOpen} className="text-xs">
			<div className="flex items-center gap-2 text-muted-foreground">
				<CollapsibleTrigger className="group flex min-w-0 items-center gap-1.5 hover:text-foreground">
					<ChevronRight className="size-3.5 shrink-0 transition-transform group-data-[state=open]:rotate-90" />
					{live && <Loader2 className="size-3.5 shrink-0 animate-spin" />}
					<span className="truncate">{activitySummary(steps)}</span>
				</CollapsibleTrigger>
				{sources > 0 && (
					<button
						type="button"
						onClick={onSources}
						className="shrink-0 underline-offset-2 hover:text-foreground hover:underline"
					>
						· {sources === 1 ? "1 source" : `${sources} sources`}
					</button>
				)}
			</div>
			<CollapsibleContent className="mt-2 flex flex-col gap-2 border-l-2 pl-3">
				{steps.map((step) =>
					step.kind === "reasoning" ? (
						<Collapsible key={step.id}>
							<CollapsibleTrigger className="group flex items-center gap-1 text-muted-foreground hover:text-foreground">
								<ChevronRight className="size-3 transition-transform group-data-[state=open]:rotate-90" />
								Reasoning
							</CollapsibleTrigger>
							<CollapsibleContent className="mt-1 whitespace-pre-wrap border-l-2 pl-3 text-muted-foreground">
								{step.content}
							</CollapsibleContent>
						</Collapsible>
					) : (
						<ToolCallCard
							key={step.call.id}
							call={step.call}
							result={results.get(step.call.id)}
							running={running}
						/>
					),
				)}
			</CollapsibleContent>
		</Collapsible>
	);
}
