import http from "node:http";
import type { AddressInfo } from "node:net";
import { Readable } from "node:stream";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openPdf } from "./pdf";
import { proxy } from "./proxy";

// A PDF of `pages` pages, each with a content stream padded to `padding`
// bytes, so the file spans many 64 KB ranges. The page dictionaries come
// first and together, as in a real PDF: pdf.js reads every one it passes
// on the way to a page.
function paddedPdf(pages: number, padding: number): Uint8Array {
	const objects: string[] = [];
	const kids = Array.from({ length: pages }, (_, i) => `${3 + i} 0 R`);
	objects.push("<< /Type /Catalog /Pages 2 0 R >>");
	objects.push(`<< /Type /Pages /Kids [${kids.join(" ")}] /Count ${pages} >>`);
	for (let i = 0; i < pages; i++) {
		objects.push(
			`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents ${3 + pages + i} 0 R >>`,
		);
	}
	for (let i = 0; i < pages; i++) {
		const content = `0 0 1 rg 10 10 50 50 re f\n%${"x".repeat(padding)}\n`;
		objects.push(
			`<< /Length ${content.length} >>\nstream\n${content}endstream`,
		);
	}
	let pdf = "%PDF-1.7\n";
	const offsets = objects.map((object, i) => {
		const offset = pdf.length;
		pdf += `${i + 1} 0 obj\n${object}\nendobj\n`;
		return offset;
	});
	const xref = pdf.length;
	pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
	for (const offset of offsets) {
		pdf += `${String(offset).padStart(10, "0")} 00000 n \n`;
	}
	pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
	return new TextEncoder().encode(pdf);
}

type Upstream = {
	ranges: (string | null)[];
	status: number | null;
	whole?: boolean;
	// Answer `status` from this request on, counting from one.
	failFrom?: number;
	// Leave every request unanswered.
	hang?: boolean;
};

// The backend's /original answer, behind the app's proxy as the browser sees it.
async function serve(body: Uint8Array, upstream: Upstream) {
	const backend = http.createServer((req, res) => {
		const range = req.headers.range ?? null;
		upstream.ranges.push(range);
		if (upstream.hang) return;
		const failing =
			upstream.failFrom !== undefined &&
			upstream.ranges.length >= upstream.failFrom;
		if (
			upstream.status !== null &&
			(upstream.failFrom === undefined || failing)
		) {
			res.writeHead(upstream.status);
			res.end();
			return;
		}
		const match = range?.match(/^bytes=(\d+)-(\d+)$/);
		if (!match || upstream.whole) {
			res.writeHead(200, {
				"content-length": body.length,
				"accept-ranges": "bytes",
			});
			res.end(body);
			return;
		}
		const start = Number(match[1]);
		const end = Math.min(Number(match[2]), body.length - 1);
		res.writeHead(206, {
			"content-length": end - start + 1,
			"accept-ranges": "bytes",
			"content-range": `bytes ${start}-${end}/${body.length}`,
		});
		res.end(body.subarray(start, end + 1));
	});
	await new Promise<void>((done) => backend.listen(0, "127.0.0.1", done));
	const base = `http://127.0.0.1:${(backend.address() as AddressInfo).port}`;
	const app = http.createServer(async (req, res) => {
		const headers = new Headers();
		for (const [name, value] of Object.entries(req.headers)) {
			if (typeof value === "string") headers.set(name, value);
		}
		const answer = await proxy(
			new Request(`http://app${req.url}`, { headers }),
			base,
			["original"],
		);
		res.writeHead(answer.status, Object.fromEntries(answer.headers));
		if (answer.body) Readable.fromWeb(answer.body as never).pipe(res);
		else res.end();
	});
	await new Promise<void>((done) => app.listen(0, "127.0.0.1", done));
	servers.push(backend, app);
	return `http://127.0.0.1:${(app.address() as AddressInfo).port}/original`;
}

const servers: http.Server[] = [];

afterEach(() => {
	vi.unstubAllGlobals();
	for (const server of servers.splice(0)) server.close();
});

describe("openPdf", () => {
	it("reads a PDF through the proxy by byte range only", async () => {
		const body = paddedPdf(40, 30_000);
		const upstream: Upstream = { ranges: [], status: null };
		const onStale = vi.fn();
		const url = await serve(body, upstream);

		const pdf = await openPdf(url, { onStale, onError: vi.fn() });
		expect(pdf?.numPages).toBe(40);
		await pdf?.getPage(20);

		expect(upstream.ranges.every((r) => r?.startsWith("bytes="))).toBe(true);
		const read = upstream.ranges.reduce((sum, r) => {
			const [, a, b] = r?.match(/(\d+)-(\d+)/) ?? [];
			return sum + Number(b) - Number(a) + 1;
		}, 0);
		expect(read).toBeLessThan(body.length / 2);
		expect(onStale).not.toHaveBeenCalled();
		await pdf?.loadingTask.destroy();
	});

	it("reports a changed original from a later range", async () => {
		const upstream: Upstream = { ranges: [], status: null };
		const onStale = vi.fn();
		const pdf = await openPdf(await serve(paddedPdf(40, 30_000), upstream), {
			onStale,
			onError: vi.fn(),
		});
		upstream.status = 409;

		const page = await pdf?.getPage(20);
		void page?.getOperatorList();
		await vi.waitFor(() => expect(onStale).toHaveBeenCalled());
		await pdf?.loadingTask.destroy();
	});

	it("reports any other failure of a later range", async () => {
		const upstream: Upstream = { ranges: [], status: null };
		const onError = vi.fn();
		const pdf = await openPdf(await serve(paddedPdf(40, 30_000), upstream), {
			onStale: vi.fn(),
			onError,
		});
		upstream.status = 500;

		const page = await pdf?.getPage(20);
		void page?.getOperatorList();
		await vi.waitFor(() =>
			expect(onError).toHaveBeenCalledWith("The original answered 500."),
		);
		await pdf?.loadingTask.destroy();
	});

	it("reports a later range that cannot be fetched", async () => {
		const upstream: Upstream = { ranges: [], status: null };
		const onError = vi.fn();
		const pdf = await openPdf(await serve(paddedPdf(40, 30_000), upstream), {
			onStale: vi.fn(),
			onError,
		});
		for (const server of servers.splice(0)) {
			server.close();
			server.closeAllConnections();
		}

		const page = await pdf?.getPage(10);
		page?.getOperatorList().catch(() => {});
		await vi.waitFor(() => expect(onError).toHaveBeenCalledOnce());
	});

	it("says nothing of a range aborted in flight", async () => {
		const upstream: Upstream = { ranges: [], status: null };
		const onError = vi.fn();
		const controller = new AbortController();
		const pdf = await openPdf(await serve(paddedPdf(40, 30_000), upstream), {
			onStale: vi.fn(),
			onError,
			signal: controller.signal,
		});
		upstream.hang = true;
		const asked = upstream.ranges.length;

		const page = await pdf?.getPage(10);
		page?.getOperatorList().catch(() => {});
		await vi.waitFor(() => expect(upstream.ranges.length).toBe(asked + 1));
		controller.abort();
		await new Promise((done) => setTimeout(done, 100));

		expect(onError).not.toHaveBeenCalled();
	});

	it.each([
		[409, null],
		[500, "The original answered 500."],
	])(
		"settles when a range needed to open answers %i",
		async (status, message) => {
			const upstream: Upstream = { ranges: [], status, failFrom: 2 };
			const onStale = vi.fn();
			const onError = vi.fn();
			const opening = openPdf(await serve(paddedPdf(40, 30_000), upstream), {
				onStale,
				onError,
			});

			if (message) {
				await expect(opening).rejects.toThrow(message);
				expect(onError).toHaveBeenCalledWith(message);
			} else {
				await expect(opening).resolves.toBeNull();
				expect(onStale).toHaveBeenCalledOnce();
			}
		},
	);

	it("settles when opening is aborted", async () => {
		const controller = new AbortController();
		const upstream: Upstream = { ranges: [], status: null };
		const url = await serve(paddedPdf(40, 30_000), upstream);
		const realFetch = globalThis.fetch;
		vi.stubGlobal("fetch", async (input: string, init: RequestInit) => {
			if (upstream.ranges.length >= 1) controller.abort();
			return realFetch(input, init);
		});

		const opened = await openPdf(url, {
			onStale: vi.fn(),
			onError: vi.fn(),
			signal: controller.signal,
		});

		expect(opened).toBeNull();
	});

	// Aborts once the first answer's body is read, then reads it on, or fails
	// the read as an aborted fetch does.
	function abortOnRead(controller: AbortController, readOn: boolean) {
		const realFetch = globalThis.fetch;
		vi.stubGlobal("fetch", async (input: string, init: RequestInit) => {
			const response = await realFetch(input, init);
			const read = response.arrayBuffer.bind(response);
			response.arrayBuffer = async () => {
				controller.abort();
				if (readOn) return read();
				throw new DOMException("This operation was aborted", "AbortError");
			};
			return response;
		});
	}

	it("settles when aborted while reading the first answer", async () => {
		const controller = new AbortController();
		const url = await serve(paddedPdf(2, 10), { ranges: [], status: null });
		abortOnRead(controller, false);

		const opened = await openPdf(url, {
			onStale: vi.fn(),
			onError: vi.fn(),
			signal: controller.signal,
		});

		expect(opened).toBeNull();
	});

	it.each([
		["a PDF within the first range", paddedPdf(2, 10), false],
		[
			"the whole answer of a server ignoring ranges",
			paddedPdf(3, 30_000),
			true,
		],
	])("settles when aborted while opening %s", async (_, body, whole) => {
		const controller = new AbortController();
		const url = await serve(body, { ranges: [], status: null, whole });
		abortOnRead(controller, true);

		const opened = await openPdf(url, {
			onStale: vi.fn(),
			onError: vi.fn(),
			signal: controller.signal,
		});

		expect(opened).toBeNull();
	});

	it("settles when aborted before it starts", async () => {
		const url = await serve(paddedPdf(2, 10), { ranges: [], status: null });

		const opened = await openPdf(url, {
			onStale: vi.fn(),
			onError: vi.fn(),
			signal: AbortSignal.abort(),
		});

		expect(opened).toBeNull();
	});

	it("opens the whole answer of a server that ignores the range", async () => {
		const upstream: Upstream = { ranges: [], status: null, whole: true };
		const pdf = await openPdf(await serve(paddedPdf(3, 30_000), upstream), {
			onStale: vi.fn(),
			onError: vi.fn(),
		});

		expect(pdf?.numPages).toBe(3);
		await pdf?.loadingTask.destroy();
	});

	it("reports a changed original from the first range", async () => {
		const upstream: Upstream = { ranges: [], status: 409 };
		const onStale = vi.fn();

		const pdf = await openPdf(await serve(paddedPdf(1, 10), upstream), {
			onStale,
			onError: vi.fn(),
		});

		expect(pdf).toBeNull();
		expect(onStale).toHaveBeenCalledOnce();
	});

	it("fails on any other answer", async () => {
		const upstream: Upstream = { ranges: [], status: 500 };
		const url = await serve(paddedPdf(1, 10), upstream);

		await expect(
			openPdf(url, { onStale: vi.fn(), onError: vi.fn() }),
		).rejects.toThrow("The original answered 500.");
	});

	it("opens a PDF that fits in the first range from it alone", async () => {
		const upstream: Upstream = { ranges: [], status: null };
		const pdf = await openPdf(await serve(paddedPdf(2, 10), upstream), {
			onStale: vi.fn(),
			onError: vi.fn(),
		});

		expect(pdf?.numPages).toBe(2);
		expect(upstream.ranges).toHaveLength(1);
		await pdf?.loadingTask.destroy();
	});
});
