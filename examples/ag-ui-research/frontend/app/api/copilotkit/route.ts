import { HttpAgent } from "@ag-ui/client";
import {
	CopilotRuntime,
	copilotRuntimeNextJSAppRouterEndpoint,
	ExperimentalEmptyAdapter,
} from "@copilotkit/runtime";
import type { NextRequest } from "next/server";

// Connect CopilotKit to PydanticAI via HttpAgent
// The HttpAgent creates a bridge between the Next.js frontend and the Python backend
// It communicates with the server created by agent.to_ag_ui()
const runtime = new CopilotRuntime({
	agents: {
		// "research_agent" maps to the agent name used in useCoAgent() on the frontend
		research_agent: new HttpAgent({
			url: `${process.env.BACKEND_URL || "http://backend:8000"}/v1/research/stream`,
		}),
	},
});

// Service adapter for multi-agent support (empty since we only have one agent)
const serviceAdapter = new ExperimentalEmptyAdapter();

// Next.js API route handler that proxies requests between frontend and backend
export async function POST(request: NextRequest) {
	const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
		runtime,
		serviceAdapter,
		endpoint: "/api/copilotkit",
	});

	return handleRequest(request);
}
