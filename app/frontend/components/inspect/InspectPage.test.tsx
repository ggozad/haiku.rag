/** @vitest-environment jsdom */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import InspectPage from "./InspectPage";

const push = vi.fn();
let path = ["alpha"];
vi.mock("next/navigation", () => ({
	useParams: () => ({ path }),
	usePathname: () => `/inspect/${path.join("/")}`,
	useSearchParams: () => new URLSearchParams(),
	useRouter: () => ({ push, replace: vi.fn() }),
}));

let tag: string | null;
let sources: () => Response;
let calls: { url: string; method: string }[];

beforeEach(() => {
	path = ["alpha"];
	tag = null;
	sources = () => Response.json([{ source_id: "fs", database: "alpha" }]);
	calls = [];
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init?: RequestInit) => {
			calls.push({ url, method: init?.method ?? "GET" });
			const path = new URL(url, "http://app").pathname;
			if (path.endsWith("/api/ingester/sources")) return sources();
			if (path.includes("/api/ingester/databases/"))
				return Response.json({ document_id: "d1", deleted: true });
			if (path.endsWith("/databases"))
				return Response.json({
					vision: false,
					databases: [
						{
							name: "alpha",
							available: true,
							error: null,
							tag,
							documents: 1,
							chunks: 3,
						},
					],
				});
			const opened = path.match(/\/documents\/(d\d)$/)?.[1];
			if (opened)
				return Response.json({
					id: opened,
					title: "Κώδικας ΦΠΑ",
					uri: "file:///data/a.pdf",
					metadata: {},
					created_at: "2026-11-05T10:00:00Z",
					updated_at: "2026-11-05T10:00:00Z",
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
			return Response.json({ items: [], outline: [], chunks: [], total: 0 });
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

	it("offers edit controls on an ingested database", async () => {
		path = ["alpha", "d1"];
		render(<InspectPage />);
		expect(await screen.findByRole("button", { name: "Edit" })).toBeTruthy();
	});

	it.each([
		[
			"a tagged database",
			() => {
				tag = "release-1";
			},
		],
		[
			"a database outside the ingester's sources",
			() => {
				sources = () => Response.json([{ source_id: "fs", database: "beta" }]);
			},
		],
		[
			"an unreachable ingester",
			() => {
				sources = () =>
					Response.json({ error: "unreachable" }, { status: 502 });
			},
		],
	])("offers none on %s", async (_, arrange) => {
		arrange();
		path = ["alpha", "d1"];
		render(<InspectPage />);
		await screen.findByRole("heading", { name: "Κώδικας ΦΠΑ" });
		await waitFor(() =>
			expect(
				calls.some((call) => call.url.endsWith("/api/ingester/sources")),
			).toBe(true),
		);
		expect(screen.queryByRole("button", { name: "Edit" })).toBeNull();
	});

	it("returns to the list after a delete", async () => {
		path = ["alpha", "d1"];
		const user = userEvent.setup();
		render(<InspectPage />);
		await user.click(await screen.findByRole("button", { name: "Delete" }));
		await user.click(screen.getByRole("button", { name: "Delete document" }));

		await waitFor(() => expect(push).toHaveBeenCalledWith("/inspect/alpha"));
		expect(
			calls.some(
				(call) =>
					call.method === "DELETE" &&
					call.url.endsWith("/api/ingester/databases/alpha/documents/d1"),
			),
		).toBe(true);
	});
});
