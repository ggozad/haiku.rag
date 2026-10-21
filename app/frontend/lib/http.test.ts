import { describe, expect, it } from "vitest";
import { decoded } from "./http";

describe("decoded", () => {
	it("returns the body of a success", async () => {
		expect(await decoded(Response.json({ a: 1 }))).toEqual({ a: 1 });
	});

	it("reports the error a failure names", async () => {
		await expect(
			decoded(Response.json({ error: "backend down" }, { status: 503 })),
		).rejects.toThrow("backend down");
		await expect(
			decoded(Response.json({ detail: "no such job" }, { status: 404 })),
		).rejects.toThrow("no such job");
	});

	it("reports the status of a failure without a JSON body", async () => {
		await expect(
			decoded(
				new Response("<html>Bad Gateway</html>", {
					status: 502,
					statusText: "Bad Gateway",
				}),
			),
		).rejects.toThrow("502 Bad Gateway");
	});

	it("refuses a success that is not JSON", async () => {
		await expect(decoded(new Response("<html></html>"))).rejects.toThrow(
			"not JSON",
		);
	});
});
