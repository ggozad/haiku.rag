import { describe, expect, it } from "vitest";
import {
	type Check,
	counts,
	formatBytes,
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

	it("sizes the vector scan and formats bytes", () => {
		expect(scanBytes(1000, 1024)).toBe(4096000);
		expect(formatBytes(512)).toBe("512 B");
		expect(formatBytes(4096000)).toBe("3.9 MB");
		expect(formatBytes(5 * 1024 ** 5)).toBe("5120.0 TB");
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
