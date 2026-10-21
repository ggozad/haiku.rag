import type { InputContent } from "@ag-ui/client";

export const MAX_IMAGES = 8;
export const MAX_EDGE = 2048;

export interface Attachment {
	// Base64 without the data URL prefix.
	data: string;
	mimeType: string;
	name: string;
}

// The size an image is drawn at so its long edge is at most `max`.
export function fitWithin(
	width: number,
	height: number,
	max = MAX_EDGE,
): { width: number; height: number } {
	const scale = Math.min(1, max / Math.max(width, height));
	return {
		width: Math.max(1, Math.round(width * scale)),
		height: Math.max(1, Math.round(height * scale)),
	};
}

// The content of a user message: the text alone, or the text followed by the
// images as AG-UI image parts.
export function userContent(
	text: string,
	images: Attachment[],
): string | InputContent[] {
	if (!images.length) return text;
	return [
		...(text ? [{ type: "text" as const, text }] : []),
		...images.map((image) => ({
			type: "image" as const,
			source: {
				type: "data" as const,
				value: image.data,
				mimeType: image.mimeType,
			},
		})),
	];
}

// The images of a stored user message, as data URLs to show.
export function imagesOf(content: unknown): string[] {
	if (!Array.isArray(content)) return [];
	return content
		.filter((part) => part?.type === "image" && part.source?.type === "data")
		.map((part) => `data:${part.source.mimeType};base64,${part.source.value}`);
}

// Reads a file as an image and redraws it within MAX_EDGE as PNG, or JPEG for
// a photograph, so the request stays small.
export async function readImage(file: File): Promise<Attachment> {
	const bitmap = await createImageBitmap(file);
	const { width, height } = fitWithin(bitmap.width, bitmap.height);
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	canvas.getContext("2d")?.drawImage(bitmap, 0, 0, width, height);
	bitmap.close();
	const mimeType = file.type === "image/jpeg" ? "image/jpeg" : "image/png";
	const dataUrl = canvas.toDataURL(mimeType, 0.9);
	return {
		data: dataUrl.slice(dataUrl.indexOf(",") + 1),
		mimeType,
		name: file.name,
	};
}
