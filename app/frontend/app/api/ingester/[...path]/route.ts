import { ingesterHeaders, ingesterUrl, proxy } from "@/lib/proxy";

export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

async function handle(request: Request, { params }: Context) {
	const { path } = await params;
	return proxy(request, ingesterUrl(), path, ingesterHeaders());
}

export const GET = handle;
export const POST = handle;
export const DELETE = handle;
