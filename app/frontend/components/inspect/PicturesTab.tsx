"use client";

import { Button } from "@/components/ui/button";

export interface Picture {
	self_ref: string;
	caption: string | null;
	description: string | null;
	page_numbers: number[];
	chunk_id: string | null;
}

export default function PicturesTab({
	pictures,
	error,
	pictureUrl,
	onOpenChunk,
}: {
	pictures: Picture[] | undefined;
	error: string | null;
	pictureUrl: (ref: string) => string;
	onOpenChunk: (chunkId: string) => void;
}) {
	return (
		<>
			{error && <p className="text-sm text-destructive">{error}</p>}
			{pictures && pictures.length === 0 && (
				<p className="text-sm text-muted-foreground">
					This document has no pictures.
				</p>
			)}
			<div className="grid gap-4 sm:grid-cols-2">
				{pictures?.map((picture) => (
					<figure key={picture.self_ref} className="space-y-1">
						{/* biome-ignore lint/performance/noImgElement: pictures come from the backend proxy */}
						<img
							src={pictureUrl(picture.self_ref)}
							alt={picture.caption || picture.description || picture.self_ref}
							className="max-h-72 rounded border"
						/>
						<figcaption className="space-y-0.5 text-xs text-muted-foreground">
							{picture.caption && <p>{picture.caption}</p>}
							{picture.description && (
								<p className="italic">{picture.description}</p>
							)}
							{!picture.caption && !picture.description && (
								<p>No caption or description.</p>
							)}
							{picture.page_numbers.length > 0 && (
								<p>p. {picture.page_numbers.join(", ")}</p>
							)}
						</figcaption>
						{picture.chunk_id && (
							<Button
								size="sm"
								variant="outline"
								onClick={() =>
									picture.chunk_id && onOpenChunk(picture.chunk_id)
								}
							>
								Open its chunk
							</Button>
						)}
					</figure>
				))}
			</div>
		</>
	);
}
