import { describe, expect, it } from "vitest";
import {
	type Check,
	counts,
	nearestDocuments,
	ordered,
	scanBytes,
	settled,
	verdict,
} from "./doctor";

const check = (name: string, severity: Check["severity"]): Check => ({
	name,
	severity,
	message: "",
	remediation: null,
	details: [],
});

describe("doctor report", () => {
	it("counts and orders checks by severity, keeping the doctor's order", () => {
		const checks = [
			check("a", "ok"),
			check("b", "fail"),
			check("c", "warn"),
			check("d", "fail"),
		];
		expect(counts(checks)).toEqual({ ok: 1, warn: 1, fail: 2 });
		expect(ordered(checks).map((c) => c.name)).toEqual(["b", "d", "c", "a"]);
	});

	it("sizes the vector scan", () => {
		expect(scanBytes(1000, 1024)).toBe(4096000);
	});

	it("knows when a job is settled", () => {
		expect(settled(null)).toBe(true);
		const job = {
			kind: "doctor",
			status: "running" as const,
			progress: [],
			queued_at: "",
			started_at: null,
			finished_at: null,
			result: null,
			error: null,
		};
		expect(settled(job)).toBe(false);
		expect(settled({ ...job, status: "failed" })).toBe(true);
	});
});

describe("verdict", () => {
	it("leads with failures, then warnings, else says everything passed", () => {
		expect(verdict({ ok: 20, warn: 1, fail: 2 })).toEqual({
			tone: "fail",
			label: "2 checks failed, 1 warning",
		});
		expect(verdict({ ok: 20, warn: 2, fail: 0 })).toEqual({
			tone: "warn",
			label: "2 warnings",
		});
		expect(verdict({ ok: 22, warn: 0, fail: 0 })).toEqual({
			tone: "ok",
			label: "All 22 checks pass",
		});
		expect(verdict({ ok: 0, warn: 0, fail: 1 })).toEqual({
			tone: "fail",
			label: "1 check failed",
		});
	});
});

describe("nearestDocuments", () => {
	const documents = ["a", "b", "c", "d", "e"].map((id) => ({
		id,
		title: null,
		uri: `test://${id}`,
	}));
	const scores = { generation: "g2", similarity: [0.2, 1, 0.9, 0.5, 0.95] };

	it("ranks the others by the score at their index, most similar first", () => {
		expect(nearestDocuments(scores, "g2", documents, "b", 3)).toEqual([
			{ id: "e", title: null, uri: "test://e", similarity: 0.95 },
			{ id: "c", title: null, uri: "test://c", similarity: 0.9 },
			{ id: "d", title: null, uri: "test://d", similarity: 0.5 },
		]);
		expect(nearestDocuments(scores, "g2", documents, "b", 10)).toHaveLength(4);
	});

	it("ignores scores computed by another job", () => {
		expect(nearestDocuments(scores, "g3", documents, "b", 3)).toBeNull();
	});

	it("needs scores and a compared document", () => {
		expect(nearestDocuments(null, "g2", documents, "b", 3)).toBeNull();
		expect(nearestDocuments(scores, "g2", documents, "gone", 3)).toBeNull();
	});
});
