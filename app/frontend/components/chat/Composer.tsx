"use client";

import { ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export default function Composer({
	draft,
	onDraftChange,
	onSend,
	onStop,
	running,
}: {
	draft: string;
	onDraftChange: (draft: string) => void;
	onSend: (text: string) => void;
	onStop: () => void;
	running: boolean;
}) {
	const submit = () => {
		const text = draft.trim();
		if (text && !running) onSend(text);
	};
	return (
		<form
			className="mx-auto flex w-full max-w-3xl items-end gap-2 px-4 pb-4"
			onSubmit={(event) => {
				event.preventDefault();
				submit();
			}}
		>
			<Textarea
				id="chat-input"
				value={draft}
				onChange={(event) => onDraftChange(event.target.value)}
				onKeyDown={(event) => {
					if (event.key === "Enter" && !event.shiftKey) {
						event.preventDefault();
						submit();
					}
				}}
				placeholder="Ask a question"
				aria-label="Question"
				className="max-h-48 min-h-11 resize-none"
				rows={1}
			/>
			{running ? (
				<Button
					type="button"
					size="icon"
					variant="outline"
					onClick={onStop}
					aria-label="Stop"
				>
					<Square />
				</Button>
			) : (
				<Button
					type="submit"
					size="icon"
					disabled={!draft.trim()}
					aria-label="Send"
				>
					<ArrowUp />
				</Button>
			)}
		</form>
	);
}
