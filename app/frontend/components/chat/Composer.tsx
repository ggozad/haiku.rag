"use client";

import { ArrowUp, ImagePlus, Square, X } from "lucide-react";
import { type Dispatch, type SetStateAction, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { type Attachment, MAX_IMAGES, readImage } from "@/lib/images";

export default function Composer({
	draft,
	onDraftChange,
	images,
	onImagesChange,
	acceptsImages,
	onSend,
	onStop,
	running,
	blocked,
}: {
	draft: string;
	onDraftChange: (draft: string) => void;
	images: Attachment[];
	onImagesChange: Dispatch<SetStateAction<Attachment[]>>;
	acceptsImages: boolean;
	onSend: (text: string, images: Attachment[]) => void;
	onStop: () => void;
	running: boolean;
	blocked: boolean;
}) {
	const picker = useRef<HTMLInputElement>(null);
	const [problem, setProblem] = useState<string | null>(null);
	// Images still being read; the question waits for them.
	const [reading, setReading] = useState(0);

	const attach = async (files: File[]) => {
		const pictures = files.filter((file) => file.type.startsWith("image/"));
		if (!acceptsImages || !pictures.length) return;
		const room = MAX_IMAGES - images.length;
		setProblem(
			pictures.length > room
				? `At most ${MAX_IMAGES} images go with one question.`
				: null,
		);
		setReading((n) => n + 1);
		try {
			const read = await Promise.all(pictures.slice(0, room).map(readImage));
			onImagesChange((current) => [...current, ...read].slice(0, MAX_IMAGES));
		} catch {
			setProblem("One of the images could not be read.");
		} finally {
			setReading((n) => n - 1);
		}
	};

	const submit = () => {
		const text = draft.trim();
		if ((text || images.length) && !running && !blocked && !reading)
			onSend(text, images);
	};

	return (
		<form
			className="mx-auto w-full max-w-3xl space-y-2 px-4 pb-4"
			onSubmit={(event) => {
				event.preventDefault();
				submit();
			}}
			onDragOver={(event) => acceptsImages && event.preventDefault()}
			onDrop={(event) => {
				if (!acceptsImages) return;
				event.preventDefault();
				attach([...event.dataTransfer.files]);
			}}
		>
			{images.length > 0 && (
				<div className="flex flex-wrap gap-2">
					{images.map((image, index) => (
						<div key={`${image.name}-${index}`} className="relative">
							{/* biome-ignore lint/performance/noImgElement: local data URL preview */}
							<img
								src={`data:${image.mimeType};base64,${image.data}`}
								alt={image.name}
								className="size-16 rounded border object-cover"
							/>
							<button
								type="button"
								aria-label={`Remove ${image.name}`}
								className="absolute -top-1.5 -right-1.5 rounded-full border bg-background p-0.5"
								onClick={() =>
									onImagesChange((current) =>
										current.filter((_, n) => n !== index),
									)
								}
							>
								<X className="size-3" />
							</button>
						</div>
					))}
				</div>
			)}
			{problem && <p className="text-xs text-destructive">{problem}</p>}
			<div className="flex items-end gap-2">
				{acceptsImages && (
					<>
						<input
							ref={picker}
							id="image-picker"
							type="file"
							accept="image/*"
							multiple
							hidden
							onChange={(event) => {
								attach([...(event.target.files ?? [])]);
								event.target.value = "";
							}}
						/>
						<Button
							type="button"
							size="icon"
							variant="ghost"
							aria-label="Attach images"
							disabled={running || images.length >= MAX_IMAGES}
							onClick={() => picker.current?.click()}
						>
							<ImagePlus />
						</Button>
					</>
				)}
				<Textarea
					id="chat-input"
					value={draft}
					onChange={(event) => onDraftChange(event.target.value)}
					onPaste={(event) => {
						const files = [...event.clipboardData.files];
						if (
							acceptsImages &&
							files.some((f) => f.type.startsWith("image/"))
						) {
							event.preventDefault();
							attach(files);
						}
					}}
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
						disabled={
							blocked || reading > 0 || (!draft.trim() && !images.length)
						}
						aria-label="Send"
					>
						<ArrowUp />
					</Button>
				)}
			</div>
		</form>
	);
}
