"use client";

import { ChevronDown, Download, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuLabel,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { formatRelativeTime } from "@/lib/format";
import {
	createSession,
	deleteSession,
	exportSessionToMarkdown,
	getAllSessions,
	getSession,
	type StoredSession,
	setActiveSessionId,
} from "@/lib/sessionStorage";

export default function SessionMenu({
	sessionId,
	onSessionChange,
}: {
	sessionId: string;
	onSessionChange: (id: string) => void;
}) {
	const [sessions, setSessions] = useState<StoredSession[]>([]);
	const current = getSession(sessionId);

	const open = (id: string) => {
		setActiveSessionId(id);
		onSessionChange(id);
	};

	return (
		<DropdownMenu
			onOpenChange={(isOpen) => isOpen && setSessions(getAllSessions())}
		>
			<DropdownMenuTrigger asChild>
				<Button variant="ghost" size="sm" className="max-w-64">
					<span className="truncate">{current?.title ?? "New Session"}</span>
					<ChevronDown />
				</Button>
			</DropdownMenuTrigger>
			<DropdownMenuContent align="start" className="w-72">
				<DropdownMenuItem onSelect={() => open(createSession().id)}>
					<Plus />
					New session
				</DropdownMenuItem>
				{current && (
					<DropdownMenuItem onSelect={() => exportSessionToMarkdown(current)}>
						<Download />
						Export as Markdown
					</DropdownMenuItem>
				)}
				<DropdownMenuItem
					variant="destructive"
					onSelect={() => {
						deleteSession(sessionId);
						const [next] = getAllSessions();
						open(next ? next.id : createSession().id);
					}}
				>
					<Trash2 />
					Delete this session
				</DropdownMenuItem>
				<DropdownMenuSeparator />
				<DropdownMenuLabel>Sessions</DropdownMenuLabel>
				{sessions.map((session) => (
					<DropdownMenuItem
						key={session.id}
						onSelect={() => open(session.id)}
						className="flex justify-between gap-2"
					>
						<span className="truncate">{session.title}</span>
						<span className="shrink-0 text-xs text-muted-foreground">
							{formatRelativeTime(session.updatedAt, true)}
						</span>
					</DropdownMenuItem>
				))}
			</DropdownMenuContent>
		</DropdownMenu>
	);
}
