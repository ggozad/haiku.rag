"use client";

import { ChevronDown, Download, Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuLabel,
	DropdownMenuSeparator,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { formatRelativeTime } from "@/lib/format";
import {
	createSession,
	deleteSession,
	exportUrl,
	listSessions,
	type SessionSummary,
	updateSession,
} from "@/lib/sessions";

export default function SessionMenu({
	sessionId,
	title,
	running,
	onSessionChange,
	onRenamed,
	onError,
}: {
	sessionId: string;
	title: string;
	running: boolean;
	onSessionChange: (id: string) => void;
	onRenamed: (title: string) => void;
	onError: (message: string) => void;
}) {
	const [sessions, setSessions] = useState<SessionSummary[]>([]);
	const [renaming, setRenaming] = useState<string | null>(null);

	const act = (action: () => Promise<void>) => () => {
		action().catch((error: Error) => onError(error.message));
	};

	return (
		<>
			<DropdownMenu
				onOpenChange={(open) => {
					if (open)
						listSessions()
							.then(setSessions)
							.catch(() => {});
				}}
			>
				<DropdownMenuTrigger asChild>
					<Button variant="ghost" size="sm" className="max-w-64">
						<span className="truncate">{title}</span>
						<ChevronDown />
					</Button>
				</DropdownMenuTrigger>
				<DropdownMenuContent align="start" className="w-72">
					<DropdownMenuItem
						onSelect={act(async () =>
							onSessionChange((await createSession()).id),
						)}
					>
						<Plus />
						New session
					</DropdownMenuItem>
					<DropdownMenuItem
						disabled={running}
						onSelect={() => setRenaming(title)}
					>
						<Pencil />
						Rename
					</DropdownMenuItem>
					<DropdownMenuItem asChild>
						<a href={exportUrl(sessionId)} download={`${title}.md`}>
							<Download />
							Export as Markdown
						</a>
					</DropdownMenuItem>
					<DropdownMenuItem
						variant="destructive"
						disabled={running}
						onSelect={act(async () => {
							await deleteSession(sessionId);
							const [next] = await listSessions();
							onSessionChange(next ? next.id : (await createSession()).id);
						})}
					>
						<Trash2 />
						Delete this session
					</DropdownMenuItem>
					<DropdownMenuSeparator />
					<DropdownMenuLabel>Sessions</DropdownMenuLabel>
					{sessions.map((session) => (
						<DropdownMenuItem
							key={session.id}
							onSelect={() => onSessionChange(session.id)}
							className="flex justify-between gap-2"
						>
							<span className="truncate">{session.title}</span>
							<span className="shrink-0 text-xs text-muted-foreground">
								{formatRelativeTime(session.updated_at, true)}
							</span>
						</DropdownMenuItem>
					))}
				</DropdownMenuContent>
			</DropdownMenu>
			<Dialog
				open={renaming !== null}
				onOpenChange={(open) => !open && setRenaming(null)}
			>
				<DialogContent className="sm:max-w-sm">
					<form
						className="space-y-4"
						onSubmit={(event) => {
							event.preventDefault();
							const name = renaming?.trim();
							if (!name) return;
							act(async () => {
								await updateSession(sessionId, { title: name });
								onRenamed(name);
							})();
							setRenaming(null);
						}}
					>
						<DialogHeader>
							<DialogTitle>Rename session</DialogTitle>
						</DialogHeader>
						<Input
							id="session-title"
							value={renaming ?? ""}
							onChange={(event) => setRenaming(event.target.value)}
							aria-label="Session title"
						/>
						<DialogFooter>
							<Button type="submit" disabled={!renaming?.trim()}>
								Save
							</Button>
						</DialogFooter>
					</form>
				</DialogContent>
			</Dialog>
		</>
	);
}
