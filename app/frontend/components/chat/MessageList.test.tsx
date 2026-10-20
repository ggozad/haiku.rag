/** @vitest-environment jsdom */
import type { Message } from "@ag-ui/client";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import type { Evidence } from "@/lib/chat";
import MessageList from "./MessageList";

beforeAll(() => {
	Element.prototype.scrollIntoView = () => {};
});
afterEach(cleanup);

// What GET /api/sessions/{id} returns after a reload: the messages, which run
// each belongs to, and each run's evidence as the backend stored it.
const stored: {
	messages: Message[];
	message_runs: Record<string, string>;
	runs: Record<string, Evidence>;
} = {
	messages: [
		{ id: "u1", role: "user", content: "What is the boiling time?" },
		{ id: "a1", role: "assistant", content: "One minute." },
		{ id: "u2", role: "user", content: "And iodine?" },
		{ id: "a2", role: "assistant", content: "Thirty minutes." },
	],
	message_runs: { a1: "run-1", a2: "run-2" },
	runs: {
		"run-1": {
			citations: [
				{
					index: 1,
					document_id: "d1",
					source: "stanevals",
					chunk_id: "c1",
					document_uri: "doc://water",
					document_title: "Field water purification",
					page_numbers: [],
					headings: null,
					content: "Boil water for at least one minute.",
				},
			],
			program: null,
			summary: { citation_status: "grounded" },
		},
		"run-2": { citations: [], program: null, summary: {} },
	},
};

describe("MessageList", () => {
	it("renders each run's stored evidence after its answer, as the backend kept it", () => {
		render(
			<MessageList
				messages={stored.messages}
				running={false}
				evidenceByRun={stored.runs}
				messageRuns={stored.message_runs}
				hidden={new Set()}
				showSource={false}
			/>,
		);
		const answer = screen.getByText("One minute.");
		const citation = screen.getByText("Field water purification");
		const next = screen.getByText("And iodine?");
		const follows = (a: Node, b: Node) =>
			Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
		expect(follows(answer, citation)).toBe(true);
		expect(follows(citation, next)).toBe(true);
		expect(screen.getAllByText("Field water purification")).toHaveLength(1);
	});
});
