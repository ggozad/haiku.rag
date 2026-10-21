/** @vitest-environment jsdom */
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SessionMenu from "./SessionMenu";

let deletes: string[];

beforeEach(() => {
	deletes = [];
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init: RequestInit = {}) => {
			if (init.method === "DELETE") {
				deletes.push(url);
				return new Response(null, { status: 204 });
			}
			return Response.json({ sessions: [{ id: "s2", title: "Other" }] });
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("SessionMenu", () => {
	it("deletes a session only once confirmed, naming it", async () => {
		const user = userEvent.setup();
		const changed = vi.fn();
		render(
			<SessionMenu
				sessionId="s1"
				title="Water purification"
				running={false}
				onSessionChange={changed}
				onRenamed={() => {}}
				onError={() => {}}
			/>,
		);
		const remove = async () => {
			await user.click(
				screen.getByRole("button", { name: /Water purification/ }),
			);
			await user.click(await screen.findByRole("menuitem", { name: /Delete/ }));
			return screen.findByRole("dialog");
		};
		let dialog = await remove();
		expect(within(dialog).getByText(/Water purification/)).toBeTruthy();
		await user.click(within(dialog).getByRole("button", { name: "Keep it" }));
		expect(deletes).toEqual([]);

		dialog = await remove();
		await user.click(
			within(dialog).getByRole("button", { name: "Delete the session" }),
		);
		await vi.waitFor(() =>
			expect(deletes).toEqual(["/api/backend/api/sessions/s1"]),
		);
		await vi.waitFor(() => expect(changed).toHaveBeenCalledWith("s2"));
	});
});
