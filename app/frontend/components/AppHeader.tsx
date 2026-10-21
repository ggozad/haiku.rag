"use client";

import { Search } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import DatabaseStatus from "@/components/DatabaseStatus";
import IngestionChip from "@/components/IngestionChip";
import { openSearch } from "@/components/SearchOverlay";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const PAGES = [
	{ href: "/", label: "Chat" },
	{ href: "/inspect", label: "Inspector", also: "/search" },
	{ href: "/doctor", label: "Doctor" },
];

const under = (pathname: string, href: string) =>
	pathname === href || pathname.startsWith(`${href}/`);

const current = (pathname: string, page: { href: string; also?: string }) =>
	page.href === "/"
		? pathname === "/"
		: under(pathname, page.href) ||
			(page.also !== undefined && under(pathname, page.also));

export default function AppHeader() {
	const pathname = usePathname();
	// Known only in the browser, so the server's render shows Ctrl.
	const [shortcut, setShortcut] = useState("Ctrl K");
	useEffect(() => {
		if (/Mac|iPhone|iPad/.test(navigator.userAgent)) setShortcut("⌘K");
	}, []);
	return (
		<header className="flex h-12 shrink-0 items-center gap-6 border-b px-4">
			<span className="font-semibold tracking-tight">haiku.rag</span>
			<nav className="flex items-center gap-1">
				{PAGES.map((page) => (
					<Link
						key={page.href}
						href={page.href}
						className={cn(
							"rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground",
							current(pathname, page) && "bg-muted text-foreground",
						)}
					>
						{page.label}
					</Link>
				))}
			</nav>
			<div className="ml-auto flex items-center gap-3">
				<Button
					variant="outline"
					size="sm"
					className="text-muted-foreground"
					onClick={openSearch}
				>
					<Search />
					Search
					<kbd className="ml-2 font-mono text-xs">{shortcut}</kbd>
				</Button>
				<DatabaseStatus />
				<IngestionChip />
			</div>
		</header>
	);
}
