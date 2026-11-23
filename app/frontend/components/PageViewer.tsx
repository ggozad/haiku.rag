"use client";

import { useEffect, useRef, useState } from "react";
import { Loading } from "@/components/Status";
import {
	boxStyle,
	type Grounding,
	type GroundingPage,
	type Original,
	originalUrl,
	pageImageUrl,
} from "@/lib/grounding";
import { openPdf, type PDFDocumentProxy } from "@/lib/pdf";
import { useJson } from "@/lib/useJson";

// The pages a visualize answer draws on, read again when the original changed
// after it was read.
export function GroundingPages({ url }: { url: string }) {
	const [reads, setReads] = useState(0);
	const grounding = useJson<Grounding>(url, reads);
	const { data } = grounding;
	// A failed read again leaves the previous answer, whose original is gone.
	if (grounding.error)
		return <p className="text-sm text-destructive">{grounding.error}</p>;
	if (!data) return <Loading />;
	if (!data.source || !data.document_id || !data.original)
		return <p className="text-sm text-muted-foreground">{data.message}</p>;
	return (
		<PageViewer
			source={data.source}
			documentId={data.document_id}
			original={data.original}
			pages={data.pages}
			onStale={() => setReads((n) => n + 1)}
		/>
	);
}

// A document's pages from its stored original, each with its boxes drawn on
// top. A PDF renders in the browser, a page when it scrolls into view. An
// image's pages are the backend's PNG renders. `onStale` runs when the
// original changed after the boxes were read.
export default function PageViewer({
	source,
	documentId,
	original,
	pages,
	focus,
	onStale,
}: {
	source: string;
	documentId: string;
	original: Original;
	pages: GroundingPage[];
	focus?: number | null;
	onStale: () => void;
}) {
	const pdf = usePdf(source, documentId, original, onStale);
	const placed = useRef(new Map<number, HTMLDivElement>());

	useEffect(() => {
		if (focus != null) placed.current.get(focus)?.scrollIntoView();
	}, [focus]);

	return (
		<div className="space-y-3">
			{pdf.error && <p className="text-sm text-destructive">{pdf.error}</p>}
			{pages.map((page) => (
				<div
					key={page.page}
					ref={(element) => {
						if (element) placed.current.set(page.page, element);
						else placed.current.delete(page.page);
					}}
					aria-label={`Page ${page.page}`}
					role="img"
					className="relative w-full overflow-hidden rounded border bg-white"
					style={{ aspectRatio: `${page.width} / ${page.height}` }}
				>
					{original.pdf ? (
						pdf.document && <PdfPage pdf={pdf.document} page={page.page} />
					) : (
						// biome-ignore lint/performance/noImgElement: page render from the backend
						<img
							src={pageImageUrl(source, documentId, page.page, original.md5)}
							alt=""
							className="absolute inset-0 size-full"
							onError={onStale}
						/>
					)}
					{page.boxes.map((box, n) => (
						<div
							// biome-ignore lint/suspicious/noArrayIndexKey: boxes have no id
							key={n}
							data-matched={box.matched}
							className={
								box.matched
									? "absolute bg-orange-400/25 outline outline-orange-500/70"
									: "absolute bg-yellow-300/20 outline outline-orange-300/50"
							}
							style={boxStyle(box, page.width, page.height)}
						/>
					))}
				</div>
			))}
		</div>
	);
}

function usePdf(
	source: string,
	documentId: string,
	original: Original,
	onStale: () => void,
): { document: PDFDocumentProxy | null; error: string | null } {
	const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
	const [error, setError] = useState<string | null>(null);
	const stale = useRef(onStale);
	stale.current = onStale;

	useEffect(() => {
		if (!original.pdf) return;
		const controller = new AbortController();
		let opened: PDFDocumentProxy | null = null;
		setError(null);
		void (async () => {
			try {
				const response = await fetch(
					originalUrl(source, documentId, original.md5),
					{ signal: controller.signal },
				);
				if (response.status === 409) {
					stale.current();
					return;
				}
				if (!response.ok) {
					throw new Error(`The original answered ${response.status}.`);
				}
				opened = await openPdf(await response.arrayBuffer());
				if (controller.signal.aborted) void opened.loadingTask.destroy();
				else setDocument(opened);
			} catch (err) {
				if (!controller.signal.aborted)
					setError(err instanceof Error ? err.message : String(err));
			}
		})();
		return () => {
			controller.abort();
			void opened?.loadingTask.destroy();
			setDocument(null);
		};
	}, [source, documentId, original.md5, original.pdf]);

	return { document, error };
}

// One PDF page, rendered at the width it is shown once it scrolls into view.
function PdfPage({ pdf, page }: { pdf: PDFDocumentProxy; page: number }) {
	const canvas = useRef<HTMLCanvasElement>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		const element = canvas.current;
		if (!element) return;
		let task: { cancel: () => void; promise: Promise<void> } | null = null;
		let cancelled = false;
		const observer = new IntersectionObserver(async ([entry]) => {
			if (!entry?.isIntersecting || task) return;
			observer.disconnect();
			try {
				const pdfPage = await pdf.getPage(page);
				if (cancelled) return;
				const width = element.clientWidth * window.devicePixelRatio;
				const viewport = pdfPage.getViewport({
					scale: width / pdfPage.getViewport({ scale: 1 }).width,
				});
				element.width = viewport.width;
				element.height = viewport.height;
				task = pdfPage.render({ canvas: element, viewport });
				await task.promise;
			} catch (err) {
				if (cancelled || (err as Error).name === "RenderingCancelledException")
					return;
				setError(err instanceof Error ? err.message : String(err));
			}
		});
		observer.observe(element);
		return () => {
			cancelled = true;
			observer.disconnect();
			task?.cancel();
		};
	}, [pdf, page]);

	return (
		<>
			<canvas ref={canvas} className="absolute inset-0 size-full" />
			{error && (
				<p className="absolute inset-x-0 top-0 p-2 text-sm text-destructive">
					Page {page} could not be rendered: {error}
				</p>
			)}
		</>
	);
}
