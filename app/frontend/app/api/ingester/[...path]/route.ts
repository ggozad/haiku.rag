import { ingesterHeaders, ingesterUrl, proxy } from "@/lib/proxy";

export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

// The operator actions: refresh a source, retry its failed jobs, retry a job or
// a dead letter, cancel a job.
const ACTIONS: Record<string, RegExp> = {
	POST: /^(sources\/[^/]+\/(refresh|retry)|jobs\/[^/]+\/retry|dlq\/[^/]+\/retry)$/,
	DELETE: /^jobs\/[^/]+$/,
};

async function handle(request: Request, { params }: Context) {
	const { path } = await params;
	const action = ACTIONS[request.method];
	if (action && !action.test(path.join("/")))
		return new Response("Not an operator action.", { status: 405 });
	return proxy(request, ingesterUrl(), path, ingesterHeaders());
}

export const GET = handle;
export const POST = handle;
export const DELETE = handle;
