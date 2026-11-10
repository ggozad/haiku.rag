// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import Highlighted from "./Highlighted";

afterEach(cleanup);

describe("Highlighted", () => {
	it("marks YAML keys and values without changing the text", () => {
		const code = "lancedb:\n  databases:\n    nomoi: /data/db/nomoi.lancedb";
		const { container } = render(
			<Highlighted code={code} language="yaml" className="" />,
		);
		const lines = [...container.querySelectorAll("pre > div")];
		expect(lines.map((line) => line.textContent)).toEqual(code.split("\n"));
		expect(container.querySelector(".token.atrule")?.textContent).toBe(
			"lancedb",
		);
	});

	it("marks Python keywords and strings", () => {
		const { container } = render(
			<Highlighted
				code={'for d in docs:\n    print("x")'}
				language="python"
				className=""
			/>,
		);
		expect(container.querySelector(".token.keyword")?.textContent).toBe("for");
		expect(container.querySelector(".token.string")?.textContent).toBe('"x"');
	});
});
