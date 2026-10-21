/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import AskAbout from "./AskAbout";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	push.mockReset();
});

function serve(selection: () => Response) {
	vi.stubGlobal(
		"fetch",
		vi.fn(async (_url: string, init: RequestInit = {}) =>
			init.method === "PATCH"
				? selection()
				: init.method === "DELETE"
					? new Response(null, { status: 204 })
					: Response.json({ id: "s9" }, { status: 201 }),
		),
	);
}

describe("AskAbout", () => {
	it("opens the chat on a new session about the document", async () => {
		serve(() => Response.json({ id: "s9" }));
		await userEvent.setup().click(
			render(<AskAbout source="orb" documentId="d1" />).getByRole("button", {
				name: "Ask about this document",
			}),
		);
		expect(push).toHaveBeenCalledWith("/?session=s9");
	});

	it("shows why it failed and opens nothing", async () => {
		serve(() => Response.json({ error: "selection refused" }, { status: 400 }));
		await userEvent.setup().click(
			render(<AskAbout source="orb" documentId="d1" />).getByRole("button", {
				name: "Ask about this document",
			}),
		);
		await screen.findByText("selection refused");
		expect(push).not.toHaveBeenCalled();
	});
});
