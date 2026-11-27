import { withBasePath } from "./paths";

export interface Box {
	l: number;
	t: number;
	r: number;
	b: number;
	matched: boolean;
}

export interface PageSize {
	page: number;
	width: number;
	height: number;
}

export interface GroundingPage extends PageSize {
	boxes: Box[];
}

export interface Original {
	sha256: string;
	pdf: boolean;
}

// The backend's answer for chunks: their boxes by page, on the original read
// from `source`. Pages are only those with boxes.
export interface Grounding {
	source: string | null;
	document_id: string | null;
	original: Original | null;
	pages: GroundingPage[];
	message: string | null;
}

const percent = (value: number) => `${Math.round(value * 1e4) / 100}%`;

// A box with a bottom-left origin, in points on a page of `width` by
// `height`, as CSS percentages of the page from its top-left corner.
export function boxStyle(
	box: Pick<Box, "l" | "t" | "r" | "b">,
	width: number,
	height: number,
): { left: string; top: string; width: string; height: string } {
	return {
		left: percent(Math.min(box.l, box.r) / width),
		top: percent((height - Math.max(box.t, box.b)) / height),
		width: percent(Math.abs(box.r - box.l) / width),
		height: percent(Math.abs(box.t - box.b) / height),
	};
}

// Every page of a document, carrying the boxes of the grounding pages.
export function withBoxes(
	pages: PageSize[],
	grounding: GroundingPage[],
): GroundingPage[] {
	const boxes = new Map(grounding.map((page) => [page.page, page.boxes]));
	return pages.map((page) => ({ ...page, boxes: boxes.get(page.page) ?? [] }));
}

function documentPath(source: string, documentId: string): string {
	return `/api/backend/api/inspect/${encodeURIComponent(source)}/documents/${encodeURIComponent(documentId)}`;
}

export function originalUrl(
	source: string,
	documentId: string,
	sha256: string,
): string {
	return withBasePath(
		`${documentPath(source, documentId)}/original?${new URLSearchParams({ sha256 })}`,
	);
}

export function pageBoxesUrl(
	source: string,
	documentId: string,
	page: number,
): string {
	return withBasePath(
		`${documentPath(source, documentId)}/pages/${page}/boxes`,
	);
}

export function pageImageUrl(
	source: string,
	documentId: string,
	page: number,
	sha256: string,
): string {
	return withBasePath(
		`${documentPath(source, documentId)}/pages/${page}?${new URLSearchParams({ sha256 })}`,
	);
}
