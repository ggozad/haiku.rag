import { afterEach, describe, expect, it, vi } from "vitest";
import { basePathFrom, withBasePath } from "./paths";

describe("basePathFrom", () => {
	it("serves at the root when unset or empty", () => {
		expect(basePathFrom(undefined)).toBe("");
		expect(basePathFrom("")).toBe("");
	});

	it("accepts a prefix of one or more segments", () => {
		expect(basePathFrom("/rag")).toBe("/rag");
		expect(basePathFrom("/tools/rag")).toBe("/tools/rag");
	});

	it.each([
		"rag",
		"/",
		"/rag/",
		"//rag",
		"/a//b",
		"/rag?x=1",
		"/rag#x",
		"/r ag",
		"/.",
		"/rag/..",
		"/../rag",
		"/rag/%2e%2e/other",
		"/rag\\other",
	])("rejects %j", (value) => {
		expect(() => basePathFrom(value)).toThrow(
			`BASE_PATH ${JSON.stringify(value)}`,
		);
	});
});

describe("withBasePath", () => {
	afterEach(() => {
		vi.unstubAllEnvs();
	});

	it("leaves a path unchanged at the root", () => {
		vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "");
		expect(withBasePath("/api/backend/api/search")).toBe(
			"/api/backend/api/search",
		);
	});

	it("prefixes a path with the base path", () => {
		vi.stubEnv("NEXT_PUBLIC_BASE_PATH", "/rag");
		expect(withBasePath("/api/ingester")).toBe("/rag/api/ingester");
	});
});
