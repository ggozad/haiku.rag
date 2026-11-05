"use client";

import type { Message } from "@ag-ui/client";
import { ArrowDown, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
	type Evidence,
	evidenceAnchors,
	segments,
	textOf,
	toolResults,
} from "@/lib/chat";
import { imagesOf } from "@/lib/images";
import Activity from "./Activity";
import EvidenceBlock from "./EvidenceBlock";
import Markdown from "./Markdown";

export default function MessageList({
	messages,
	running,
	evidenceByRun,
	messageRuns,
	hidden,
	showSource,
	empty,
}: {
	messages: Message[];
	running: boolean;
	evidenceByRun: Record<string, Evidence>;
	messageRuns: Record<string, string>;
	// User-role messages a run inserted itself, which the person did not type.
	hidden: Set<string>;
	showSource: boolean;
	empty: string;
}) {
	const bottom = useRef<HTMLDivElement>(null);
	const region = useRef<HTMLDivElement>(null);
	// Whether the reader is at the bottom, so a growing answer is followed.
	const [following, setFollowing] = useState(true);
	const lastRole = messages.at(-1)?.role;
	const results = toolResults(messages);
	const anchors = running ? new Map() : evidenceAnchors(messages, messageRuns);
	const parts = segments(messages);
	const lastContent = messages.length
		? JSON.stringify(messages[messages.length - 1]).length
		: 0;

	const jump = () => {
		bottom.current?.scrollIntoView({ block: "end" });
		setFollowing(true);
	};

	// A question the reader sends brings them back to the bottom.
	// biome-ignore lint/correctness/useExhaustiveDependencies: a new message decides
	useEffect(() => {
		if (lastRole === "user") jump();
	}, [messages.length]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: scroll as the last message grows and when a finished run's evidence appears
	useEffect(() => {
		if (following) bottom.current?.scrollIntoView({ block: "end" });
	}, [messages.length, lastContent, running]);

	return (
		<div
			ref={region}
			role="log"
			className="relative min-h-0 flex-1 overflow-y-auto"
			onScroll={() => {
				const el = region.current;
				if (el)
					setFollowing(el.scrollHeight - el.scrollTop - el.clientHeight < 64);
			}}
		>
			<div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6">
				{messages.length === 0 && (
					<p className="py-24 text-center text-base text-muted-foreground">
						{empty}
					</p>
				)}
				{parts.map((part, position) => {
					if (part.kind === "activity") {
						const runId = part.messageIds
							.map((id) => messageRuns[id])
							.find(Boolean);
						const evidence = runId ? evidenceByRun[runId] : undefined;
						return (
							<Activity
								key={part.key}
								steps={part.steps}
								results={results}
								live={running && position === parts.length - 1}
								running={running}
								sources={running ? 0 : (evidence?.citations.length ?? 0)}
								onSources={() =>
									document
										.getElementById(`evidence-${runId}`)
										?.scrollIntoView({ block: "start", behavior: "smooth" })
								}
							/>
						);
					}
					const { message, index } = part;
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
							{message.content && <Markdown>{message.content}</Markdown>}
							{anchors.has(index) && evidenceByRun[anchors.get(index)] && (
								<div
									id={`evidence-${anchors.get(index)}`}
									className="scroll-mt-4"
								>
									<EvidenceBlock
										evidence={evidenceByRun[anchors.get(index)]}
										showSource={showSource}
									/>
								</div>
							)}
						</div>
					);
				})}
				{running && parts.at(-1)?.kind !== "activity" && (
					<Loader2 className="size-4 animate-spin text-muted-foreground" />
				)}
				<div ref={bottom} />
			</div>
			{!following && (
				<div className="pointer-events-none sticky bottom-3 flex justify-center">
					<Button
						size="sm"
						variant="secondary"
						className="pointer-events-auto shadow"
						onClick={jump}
					>
						<ArrowDown />
						Jump to latest
					</Button>
				</div>
			)}
		</div>
	);
}
