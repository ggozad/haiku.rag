import { HttpAgent } from "@ag-ui/client";
import {
	CopilotRuntime,
	copilotRuntimeNextJSAppRouterEndpoint,
	ExperimentalEmptyAdapter,
} from "@copilotkit/runtime";
import type { NextRequest } from "next/server";

const runtime = new CopilotRuntime({
	agents: {
		chat_agent: new HttpAgent({
			url: `${process.env.BACKEND_URL || "http://backend:8000"}/v1/chat/stream`,
		}),
	},
});

const serviceAdapter = new ExperimentalEmptyAdapter();

export async function POST(request: NextRequest) {
	const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
		runtime,
		serviceAdapter,
		endpoint: "/api/copilotkit",
	});

	return handleRequest(request);
}
