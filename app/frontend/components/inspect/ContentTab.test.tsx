/** @vitest-environment jsdom */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { blocksOf, type Item } from "@/lib/inspect";
import ContentTab from "./ContentTab";

afterEach(cleanup);

const table: Item = {
	self_ref: "#/tables/0",
	label: "table",
	text: "| Rate | Applies to |\n|------|------------|\n| 13%  | Food       |",
	page_numbers: [1],
	heading_level: 0,
	chunk_ids: ["c1"],
};

describe("ContentTab", () => {
	it("renders a table item as a table", () => {
		render(
			<ContentTab
				items={[table]}
				blocks={blocksOf([table])}
				loading={false}
				error={null}
				pictureUrl={() => ""}
				highlighted={null}
				onOpenChunk={() => {}}
			/>,
		);
		expect(screen.getByRole("columnheader", { name: "Rate" })).toBeTruthy();
		expect(screen.getByRole("cell", { name: "Food" })).toBeTruthy();
	});
});
