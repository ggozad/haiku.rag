"use client";

import type { PDFDocumentProxy } from "pdfjs-dist";
import { withBasePath } from "./paths";

export type { PDFDocumentProxy };

// Bytes per range request, pdf.js's own default.
const RANGE_BYTES = 1 << 16;

export interface PdfLoading {
	// The original answered 409: it changed since its address was read.
	onStale: () => void;
	// A later range failed otherwise.
	onError: (message: string) => void;
	signal?: AbortSignal;
}

// Opens the PDF at `url` with pdf.js's legacy build, which polyfills what
// Safari 18 and Chrome 125 lack, reading it by byte range: pdf.js asks for the
// ranges a page needs and this fetches them, so a 409 on any range reaches
// `onStale`. pdf.js's own loader leaves a failed later range pending. A range
// that fails or is aborted ends the loading task, so opening always settles:
// to null on a 409 or an abort.
export async function openPdf(
	url: string,
	{ onStale, onError, signal }: PdfLoading,
): Promise<PDFDocumentProxy | null> {
	const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
	// The worker is emitted by the bundler. Under node, pdf.js runs it in-process.
	if (typeof Worker !== "undefined") {
		pdfjs.GlobalWorkerOptions.workerPort ??= new Worker(
			new URL("pdfjs-dist/legacy/build/pdf.worker.min.mjs", import.meta.url),
			{ type: "module" },
		);
	}
	// The image decoders are copied to public/pdfjs/ by the build.
	const options = { wasmUrl: withBasePath("/pdfjs/") };

	let first: Response;
	let initial: Uint8Array;
	try {
		first = await fetchRange(url, 0, RANGE_BYTES, signal);
		if (first.status === 409) {
			onStale();
			return null;
		}
		if (!first.ok) throw new Error(`The original answered ${first.status}.`);
		initial = new Uint8Array(await first.arrayBuffer());
	} catch (err) {
		if (signal?.aborted) return null;
		throw err;
	}
	const length = Number(first.headers.get("content-range")?.split("/")[1]);
	const transport =
		first.status === 200 || initial.length >= length
			? null
			: new pdfjs.PDFDataRangeTransport(length, initial);

	let ended: (error: Error | null) => void = () => {};
	const ending = new Promise<Error | null>((settle) => {
		ended = settle;
	});
	const task = pdfjs.getDocument(
		transport
			? {
					...options,
					range: transport,
					rangeChunkSize: RANGE_BYTES,
					disableAutoFetch: true,
					disableStream: true,
				}
			: { ...options, data: initial },
	);
	// Destroying the task leaves its promise pending.
	task.promise.catch(() => {});
	const finish = (error: Error | null) => {
		ended(error);
		void task.destroy();
	};
	if (signal?.aborted) finish(null);
	signal?.addEventListener("abort", () => finish(null), { once: true });
	if (transport) {
		transport.requestDataRange = (begin: number, end: number) => {
			void (async () => {
				try {
					const response = await fetchRange(url, begin, end, signal);
					if (response.status === 409) {
						onStale();
						finish(null);
					} else if (!response.ok) {
						const message = `The original answered ${response.status}.`;
						onError(message);
						finish(new Error(message));
					} else
						transport.onDataRange(
							begin,
							new Uint8Array(await response.arrayBuffer()),
						);
				} catch (err) {
					if (signal?.aborted) return;
					const error = err instanceof Error ? err : new Error(String(err));
					onError(error.message);
					finish(error);
				}
			})();
		};
	}
	const outcome = await Promise.race([
		task.promise,
		ending.then((error) => ({ error })),
	]);
	if (!("error" in outcome)) return outcome;
	if (outcome.error) throw outcome.error;
	return null;
}

// The bytes from `begin` up to, not including, `end`.
function fetchRange(
	url: string,
	begin: number,
	end: number,
	signal?: AbortSignal,
): Promise<Response> {
	return fetch(url, {
		headers: { Range: `bytes=${begin}-${end - 1}` },
		signal,
	});
}
