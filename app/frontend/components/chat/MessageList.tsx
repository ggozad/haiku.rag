"use client";

import type { Message } from "@ag-ui/client";
import { ChevronRight, Loader2 } from "lucide-react";
import { useEffect, useRef } from "react";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
	type Evidence,
	evidenceAnchors,
	textOf,
	toolResults,
} from "@/lib/chat";
import { imagesOf } from "@/lib/images";
import EvidenceBlock from "./EvidenceBlock";
import Markdown from "./Markdown";
import ToolCallCard from "./ToolCallCard";

export default function MessageList({
	messages,
	running,
	evidenceByRun,
	messageRuns,
	hidden,
	showSource,
}: {
	messages: Message[];
	running: boolean;
	evidenceByRun: Record<string, Evidence>;
	messageRuns: Record<string, string>;
	// User-role messages a run inserted itself, which the person did not type.
	hidden: Set<string>;
	showSource: boolean;
}) {
	const bottom = useRef<HTMLDivElement>(null);
	const results = toolResults(messages);
	const anchors = running ? new Map() : evidenceAnchors(messages, messageRuns);
	const lastContent = messages.length
		? JSON.stringify(messages[messages.length - 1]).length
		: 0;

	// biome-ignore lint/correctness/useExhaustiveDependencies: scroll as the last message grows
	useEffect(() => {
		bottom.current?.scrollIntoView({ block: "end" });
	}, [messages.length, lastContent]);

	return (
		<div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6">
			{messages.length === 0 && (
				<p className="py-16 text-center text-sm text-muted-foreground">
					Ask a question about the documents.
				</p>
			)}
			{messages.map((message, index) => {
				if (message.role === "reasoning") {
					return (
						<Collapsible key={message.id} className="text-xs">
							<CollapsibleTrigger className="group flex items-center gap-1 text-muted-foreground">
								<ChevronRight className="size-3 transition-transform group-data-[state=open]:rotate-90" />
								Reasoning
							</CollapsibleTrigger>
							<CollapsibleContent className="mt-1 whitespace-pre-wrap border-l-2 pl-3 text-muted-foreground">
								{message.content}
							</CollapsibleContent>
						</Collapsible>
					);
				}
				if (message.role === "user" && hidden.has(message.id)) return null;
				if (message.role === "user") {
					return (
						<div
							key={message.id}
							className="flex max-w-[85%] flex-col items-end gap-2 self-end"
						>
							{imagesOf(message.content).length > 0 && (
								<div className="flex flex-wrap justify-end gap-2">
									{imagesOf(message.content).map((src, n) => (
										// biome-ignore lint/performance/noImgElement: data URL from the message
										<img
											// biome-ignore lint/suspicious/noArrayIndexKey: images have no id
											key={n}
											src={src}
											alt={`Attachment ${n + 1}`}
											className="max-h-40 rounded border"
										/>
									))}
								</div>
							)}
							{textOf(message) && (
								<div className="whitespace-pre-wrap rounded-2xl bg-muted px-4 py-2 text-sm">
									{textOf(message)}
								</div>
							)}
						</div>
					);
				}
				if (message.role !== "assistant") return null;
				return (
					<div key={message.id} className="flex flex-col gap-2">
						{message.toolCalls?.map((call) => (
							<ToolCallCard
								key={call.id}
								call={call}
								result={results.get(call.id)}
								running={running}
							/>
						))}
						{message.content && <Markdown>{message.content}</Markdown>}
						{anchors.has(index) && evidenceByRun[anchors.get(index)] && (
							<EvidenceBlock
								evidence={evidenceByRun[anchors.get(index)]}
								showSource={showSource}
							/>
						)}
					</div>
				);
			})}
			{running && (
				<Loader2 className="size-4 animate-spin text-muted-foreground" />
			)}
			<div ref={bottom} />
		</div>
	);
}
