/** @vitest-environment jsdom */
import type { Message } from "@ag-ui/client";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Evidence } from "@/lib/chat";
import MessageList from "./MessageList";

let scrolls: number;
beforeEach(() => {
	scrolls = 0;
	Element.prototype.scrollIntoView = () => {
		scrolls++;
	};
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

	it("links a citation to its chunk's page in a new tab", async () => {
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
		await userEvent.setup().click(screen.getByText("Field water purification"));
		const link = screen.getByRole("link", { name: "Open in the inspector" });
		expect(link.getAttribute("href")).toBe("/inspect/stanevals/d1/chunks/c1");
		expect(link.getAttribute("target")).toBe("_blank");
	});

	it("follows a streaming answer only while the reader is at the bottom", async () => {
		const list = (last: string, extra: Message[] = []) => (
			<MessageList
				messages={[
					{ id: "u1", role: "user", content: "Q" },
					{ id: "a1", role: "assistant", content: last },
					...extra,
				]}
				running
				evidenceByRun={{}}
				messageRuns={{}}
				hidden={new Set()}
				showSource={false}
			/>
		);
		const { rerender } = render(list("One"));
		const region = screen.getByRole("log");
		Object.defineProperties(region, {
			scrollHeight: { configurable: true, value: 2000 },
			clientHeight: { configurable: true, value: 500 },
			scrollTop: { configurable: true, writable: true, value: 1500 },
		});
		rerender(list("One two"));
		const atBottom = scrolls;
		expect(atBottom).toBeGreaterThan(0);

		region.scrollTop = 200;
		fireEvent.scroll(region);
		rerender(list("One two three"));
		expect(scrolls).toBe(atBottom);
		const jump = screen.getByRole("button", { name: "Jump to latest" });

		fireEvent.click(jump);
		expect(scrolls).toBe(atBottom + 1);
		expect(screen.queryByRole("button", { name: "Jump to latest" })).toBeNull();

		region.scrollTop = 200;
		fireEvent.scroll(region);
		rerender(
			list("One two three", [{ id: "u2", role: "user", content: "Next" }]),
		);
		expect(scrolls).toBe(atBottom + 2);
	});
});
