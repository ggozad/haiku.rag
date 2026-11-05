/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import InspectPage from "./InspectPage";

const push = vi.fn();
vi.mock("next/navigation", () => ({
	useParams: () => ({ path: ["alpha"] }),
	usePathname: () => "/inspect/alpha",
	useSearchParams: () => new URLSearchParams(),
	useRouter: () => ({ push, replace: vi.fn() }),
}));

beforeEach(() => {
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string) => {
			const path = new URL(url, "http://app").pathname;
			if (path.endsWith("/databases"))
				return Response.json({
					vision: false,
					databases: [
						{
							name: "alpha",
							available: true,
							error: null,
							documents: 1,
							chunks: 3,
						},
					],
				});
			if (path.endsWith("/documents"))
				return Response.json({
					total: 1,
					documents: [
						{
							id: "d1",
							title: "Κώδικας ΦΠΑ",
							uri: "file:///data/sources/codifications/%CE%A1%CE%A8.pdf",
							source_id: "codifications",
							content_type: "application/pdf",
							created_at: "2026-11-05T10:00:00Z",
							updated_at: "2026-11-05T10:00:00Z",
						},
					],
				});
			return Response.json({});
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	push.mockReset();
});

describe("InspectPage", () => {
	it("lists a document by its title, source and decoded file name", async () => {
		render(<InspectPage />);
		const title = await screen.findByText("Κώδικας ΦΠΑ");
		const row = title.closest("button");
		expect(row?.textContent).toContain("codifications");
		expect(row?.textContent).toContain("ΡΨ.pdf");
		expect(row?.textContent).not.toContain("%CE");
	});

	it("filters the list by title on Enter", async () => {
		const user = userEvent.setup();
		render(<InspectPage />);
		await screen.findByText("Κώδικας ΦΠΑ");
		await user.type(
			screen.getByPlaceholderText("Filter by title or URI"),
			"ΦΠΑ{Enter}",
		);
		expect(push).toHaveBeenCalledWith(
			`/inspect/alpha?${new URLSearchParams({ list_q: "ΦΠΑ" })}`,
		);
	});

	it("keeps the SQL filter folded until asked for", async () => {
		render(<InspectPage />);
		await screen.findByText("Κώδικας ΦΠΑ");
		const sql = screen.getByPlaceholderText(/SQL filter/);
		expect(sql.closest("details")?.open).toBe(false);
	});

	it("opens the search page from its tab", async () => {
		render(<InspectPage />);
		await screen.findByText("Κώδικας ΦΠΑ");
		await userEvent
			.setup()
			.click(screen.getByRole("button", { name: "Search" }));
		expect(push).toHaveBeenCalledWith("/search");
	});
});
