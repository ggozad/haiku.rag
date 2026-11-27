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
					time_range: [4.48, 10.001],
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
				empty=""
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
		expect(screen.getByText("00:00:04-00:00:10")).toBeTruthy();
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
				empty=""
			/>,
		);
		await userEvent.setup().click(screen.getByText("Field water purification"));
		const link = screen.getByRole("link", { name: "Open in the inspector" });
		expect(link.getAttribute("href")).toBe("/inspect/stanevals/d1/chunks/c1");
		expect(link.getAttribute("target")).toBe("_blank");
	});

	it("links a citation to its document's web address", async () => {
		const [run] = Object.values(stored.runs);
		const evidence = {
			...run,
			citations: [
				{
					...run.citations[0],
					document_meta: { url: "https://example.org/water" },
				},
			],
		};
		render(
			<MessageList
				messages={stored.messages}
				running={false}
				evidenceByRun={{ ...stored.runs, "run-1": evidence }}
				messageRuns={stored.message_runs}
				hidden={new Set()}
				showSource={false}
				empty=""
			/>,
		);
		await userEvent.setup().click(screen.getByText("Field water purification"));
		const link = screen.getByRole("link", { name: "example.org" });
		expect(link.getAttribute("href")).toBe("https://example.org/water");
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
				empty=""
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

	it("shows a run's steps while it works and folds them once the answer is in", async () => {
		const messages: Message[] = [
			{ id: "u1", role: "user", content: "What is the boiling time?" },
			{ id: "r1", role: "reasoning", content: "Look it up." },
			{
				id: "a0",
				role: "assistant",
				content: "",
				toolCalls: [
					{
						id: "c1",
						type: "function",
						function: {
							name: "search",
							arguments: JSON.stringify({ query: "boiling" }),
						},
					},
				],
			},
			{ id: "t1", role: "tool", toolCallId: "c1", content: "found" },
			{ id: "a1", role: "assistant", content: "One minute." },
		];
		const props = {
			messages,
			evidenceByRun: stored.runs,
			messageRuns: { a0: "run-1", a1: "run-1" },
			hidden: new Set<string>(),
			showSource: false,
			empty: "",
		};
		const { rerender } = render(
			<MessageList {...props} messages={messages.slice(0, -1)} running />,
		);
		expect(screen.getByText("boiling")).toBeTruthy();

		rerender(<MessageList {...props} running={false} />);
		expect(screen.getByText("1 search")).toBeTruthy();
		expect(screen.queryByText("boiling")).toBeNull();

		await userEvent.setup().click(screen.getByText("1 search"));
		expect(screen.getByText("boiling")).toBeTruthy();
	});

	it("jumps from the summary line to the run's sources", async () => {
		const messages: Message[] = [
			{ id: "u1", role: "user", content: "Q" },
			{
				id: "a0",
				role: "assistant",
				content: "",
				toolCalls: [
					{
						id: "c1",
						type: "function",
						function: { name: "search", arguments: "{}" },
					},
				],
			},
			{ id: "a1", role: "assistant", content: "One minute." },
		];
		render(
			<MessageList
				messages={messages}
				running={false}
				evidenceByRun={stored.runs}
				messageRuns={{ a0: "run-1", a1: "run-1" }}
				hidden={new Set()}
				showSource={false}
				empty=""
			/>,
		);
		const before = scrolls;
		await userEvent
			.setup()
			.click(screen.getByRole("button", { name: "· 1 source" }));
		expect(scrolls).toBe(before + 1);
	});

	it("scrolls to the end when a finished run's evidence appears", () => {
		const props = {
			messages: stored.messages.slice(0, 2),
			evidenceByRun: stored.runs,
			messageRuns: stored.message_runs,
			hidden: new Set<string>(),
			showSource: false,
			empty: "",
		};
		const { rerender } = render(<MessageList {...props} running />);
		const before = scrolls;
		rerender(<MessageList {...props} running={false} />);
		expect(scrolls).toBeGreaterThan(before);
	});
});
