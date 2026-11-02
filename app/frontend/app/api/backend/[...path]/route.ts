import { backendUrl, proxy } from "@/lib/proxy";

export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

async function handle(request: Request, { params }: Context) {
	const { path } = await params;
	// An operator action: the backend serves it, the browser does not reach it.
	if (path.join("/") === "reload")
		return new Response("Not a browser action.", { status: 405 });
	return proxy(request, backendUrl(), path);
}

export const GET = handle;
export const POST = handle;
export const PUT = handle;
export const PATCH = handle;
export const DELETE = handle;
