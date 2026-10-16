import type { ToolCall } from "@ag-ui/client";
import { Check, Code, Loader2, Quote, Search, X } from "lucide-react";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";

function parseArgs(raw: string): Record<string, unknown> {
	try {
		const parsed = JSON.parse(raw);
		return parsed && typeof parsed === "object" ? parsed : {};
	} catch {
		return {};
	}
}

const LABELS: Record<string, { label: string; icon: typeof Search }> = {
	search: { label: "Search", icon: Search },
	execute_code: { label: "Code", icon: Code },
	cite: { label: "Cite", icon: Quote },
};

export default function ToolCallCard({
	call,
	result,
	running,
}: {
	call: ToolCall;
	result: string | undefined;
	running: boolean;
}) {
	const args = parseArgs(call.function.arguments);
	const { label, icon: Icon } = LABELS[call.function.name] ?? {
		label: call.function.name,
		icon: Code,
	};
	const status =
		result !== undefined ? (
			<Check className="size-3.5 text-muted-foreground" />
		) : running ? (
			<Loader2 className="size-3.5 animate-spin text-muted-foreground" />
		) : (
			<X className="size-3.5 text-muted-foreground" />
		);

	let summary: React.ReactNode = null;
	if (call.function.name === "search") {
		const sources = Array.isArray(args.sources) ? args.sources.join(", ") : "";
		summary = (
			<span className="truncate">
				{String(args.query ?? "")}
				{sources && (
					<span className="text-muted-foreground"> in {sources}</span>
				)}
			</span>
		);
	} else if (call.function.name === "cite") {
		const ids = Array.isArray(args.chunk_ids) ? args.chunk_ids.length : 0;
		summary = <span>{ids === 1 ? "1 passage" : `${ids} passages`}</span>;
	}

	const header = (
		<div className="flex min-w-0 items-center gap-2 text-xs">
			{status}
			<Icon className="size-3.5 shrink-0 text-muted-foreground" />
			<span className="font-medium">{label}</span>
			{summary}
		</div>
	);

	if (call.function.name !== "execute_code") {
		return (
			<div className="rounded-md border bg-muted/40 px-3 py-2">{header}</div>
		);
	}
	return (
		<Collapsible className="rounded-md border bg-muted/40 px-3 py-2">
			<CollapsibleTrigger className="w-full text-left">
				{header}
			</CollapsibleTrigger>
			<CollapsibleContent>
				<pre className="mt-2 overflow-x-auto rounded bg-background p-2 font-mono text-xs">
					{String(args.code ?? "")}
				</pre>
				{result !== undefined && (
					<pre className="mt-2 max-h-64 overflow-auto rounded bg-background p-2 font-mono text-xs text-muted-foreground">
						{result}
					</pre>
				)}
			</CollapsibleContent>
		</Collapsible>
	);
}
