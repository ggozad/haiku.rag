/** @vitest-environment jsdom */
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";
import DatabaseStatus from "./DatabaseStatus";

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const listing = (available: boolean) => ({
	vision: false,
	databases: [
		{
			name: "alpha",
			available,
			error: available ? null : "not created yet",
			documents: 3,
			chunks: 9,
		},
	],
});

describe("DatabaseStatus", () => {
	it("shows a database that becomes available without a reload", async () => {
		const answers = [listing(false), listing(true)];
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => Response.json(answers.shift() ?? listing(true))),
		);
		render(
			<TooltipProvider>
				<DatabaseStatus />
			</TooltipProvider>,
		);
		const badge = () => screen.getByText("alpha").closest("[data-variant]");
		await waitFor(() =>
			expect(badge()?.getAttribute("data-variant")).toBe("destructive"),
		);
		act(() => {
			window.dispatchEvent(new Event("focus"));
		});
		await waitFor(() =>
			expect(badge()?.getAttribute("data-variant")).toBe("secondary"),
		);
	});

	it("shows a database going away, and the backend failing and recovering", async () => {
		const answers: (() => Promise<Response>)[] = [
			async () => Response.json(listing(true)),
			async () => Response.json(listing(false)),
			async () => Response.json({ error: "backend down" }, { status: 503 }),
			async () => Response.json(listing(true)),
		];
		vi.stubGlobal(
			"fetch",
			vi.fn(() => (answers.shift() as () => Promise<Response>)()),
		);
		render(
			<TooltipProvider>
				<DatabaseStatus />
			</TooltipProvider>,
		);
		const variant = () =>
			screen
				.queryByText("alpha")
				?.closest("[data-variant]")
				?.getAttribute("data-variant");
		const refresh = () =>
			act(() => {
				window.dispatchEvent(new Event("focus"));
			});
		await waitFor(() => expect(variant()).toBe("secondary"));
		refresh();
		await waitFor(() => expect(variant()).toBe("destructive"));
		refresh();
		await screen.findByText("Backend unavailable");
		refresh();
		await waitFor(() => expect(variant()).toBe("secondary"));
		expect(screen.queryByText("Backend unavailable")).toBeNull();
	});

	it("shows the counts of the latest read", async () => {
		const counted = (documents: number) => ({
			vision: false,
			databases: [
				{ name: "alpha", available: true, error: null, documents, chunks: 2 },
			],
		});
		const answers = [counted(3), counted(5)];
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => Response.json(answers.shift() ?? counted(5))),
		);
		const user = userEvent.setup();
		render(
			<TooltipProvider delayDuration={0}>
				<DatabaseStatus />
			</TooltipProvider>,
		);
		await user.hover(await screen.findByText("alpha"));
		expect(
			(await screen.findAllByText("3 documents, 2 chunks")).length,
		).toBeGreaterThan(0);
		act(() => {
			window.dispatchEvent(new Event("focus"));
		});
		expect(
			(await screen.findAllByText("5 documents, 2 chunks")).length,
		).toBeGreaterThan(0);
	});
});
