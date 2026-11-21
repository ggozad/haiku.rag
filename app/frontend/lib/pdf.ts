"use client";

import type { PDFDocumentProxy } from "pdfjs-dist";
import { withBasePath } from "./paths";

export type { PDFDocumentProxy };

// Opens a PDF with pdf.js's legacy build, which polyfills what Safari 18 and
// Chrome 125 lack. The worker is emitted by the bundler, and the image
// decoders are copied to public/pdfjs/ by the build.
export async function openPdf(data: ArrayBuffer): Promise<PDFDocumentProxy> {
	const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
	pdfjs.GlobalWorkerOptions.workerPort ??= new Worker(
		new URL("pdfjs-dist/legacy/build/pdf.worker.min.mjs", import.meta.url),
		{ type: "module" },
	);
	return pdfjs.getDocument({ data, wasmUrl: withBasePath("/pdfjs/") }).promise;
}
