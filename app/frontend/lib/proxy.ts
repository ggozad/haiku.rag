// Headers that describe one hop, or that the upstream fetch sets itself.
const REQUEST_HEADERS_DROPPED = [
	"connection",
	"content-length",
	"host",
	"keep-alive",
	"transfer-encoding",
];
// fetch has already decoded the upstream body.
const RESPONSE_HEADERS_DROPPED = [
	"connection",
	"content-encoding",
	"content-length",
	"keep-alive",
	"transfer-encoding",
];

// Forwards a request to `base` + `path`, streaming both bodies. The incoming
// request's abort signal is passed on, so a browser that stops reading closes
// the upstream request as well.
export async function proxy(
	request: Request,
	base: string,
	path: string[],
	extraHeaders: Record<string, string> = {},
): Promise<Response> {
	const target = `${base.replace(/\/+$/, "")}/${path
		.map(encodeURIComponent)
		.join("/")}${new URL(request.url).search}`;
	const headers = new Headers(request.headers);
	for (const name of REQUEST_HEADERS_DROPPED) headers.delete(name);
	for (const [name, value] of Object.entries(extraHeaders)) {
		headers.set(name, value);
	}
	const hasBody = request.method !== "GET" && request.method !== "HEAD";

	let upstream: Response;
	try {
		upstream = await fetch(target, {
			method: request.method,
			headers,
			body: hasBody ? request.body : undefined,
			// Required by Node's fetch to send a streamed request body.
			duplex: "half",
			signal: request.signal,
			redirect: "manual",
		} as RequestInit);
	} catch (error) {
		if (request.signal.aborted) {
			return new Response(null, { status: 499 });
		}
		const reason = error instanceof Error ? error.message : String(error);
		return Response.json(
			{ error: `${base} is unreachable: ${reason}` },
			{ status: 502 },
		);
	}

	const responseHeaders = new Headers(upstream.headers);
	for (const name of RESPONSE_HEADERS_DROPPED) responseHeaders.delete(name);
	return new Response(upstream.body, {
		status: upstream.status,
		statusText: upstream.statusText,
		headers: responseHeaders,
	});
}

export function backendUrl(): string {
	return process.env.BACKEND_URL || "http://backend:8000";
}

export function ingesterUrl(): string {
	return process.env.INGESTER_URL || "http://ingester:8765";
}

export function ingesterHeaders(): Record<string, string> {
	const token = process.env.INGESTER_TOKEN;
	return token ? { authorization: `Bearer ${token}` } : {};
}
