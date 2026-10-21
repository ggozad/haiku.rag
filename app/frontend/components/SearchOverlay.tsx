"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import SearchPanel from "@/components/inspect/SearchPanel";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
	chunkHref,
	type SearchInputs,
	searchHref,
	searchOf,
} from "@/lib/addresses";
import { useDatabases } from "@/lib/databases";

let opener: (() => void) | null = null;

// Opens the search overlay from anywhere on the page.
export function openSearch() {
	opener?.();
}

// Search over every database, opened with Ctrl+K or Cmd+K. It keeps the last
// search while the page stays open.
export default function SearchOverlay() {
	const router = useRouter();
	const { info } = useDatabases();
	const available = info?.databases.filter((db) => db.available) ?? [];
	const [open, setOpen] = useState(false);
	const [inputs, setInputs] = useState<SearchInputs>(() =>
		searchOf(new URLSearchParams()),
	);

	useEffect(() => {
		opener = () => setOpen(true);
		const onKey = (event: KeyboardEvent) => {
			if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
				event.preventDefault();
				setOpen(true);
			}
		};
		window.addEventListener("keydown", onKey);
		return () => {
			opener = null;
			window.removeEventListener("keydown", onKey);
		};
	}, []);

	const href = searchHref(inputs);
	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogContent
				showCloseButton={false}
				className="flex h-[85dvh] flex-col gap-0 p-0 sm:max-w-5xl"
			>
				<DialogTitle className="sr-only">Search the documents</DialogTitle>
				<SearchPanel
					key={href}
					databases={available}
					inputs={inputs}
					onSearch={setInputs}
					onOpen={(ref) => {
						setOpen(false);
						router.push(chunkHref(ref.source, ref.documentId, ref.chunkId));
					}}
				/>
				{inputs.dbs.length > 0 && (
					<div className="border-t px-4 py-2 text-right text-xs">
						<Link
							href={href}
							className="underline"
							onClick={() => setOpen(false)}
						>
							Open as a page
						</Link>
					</div>
				)}
			</DialogContent>
		</Dialog>
	);
}
