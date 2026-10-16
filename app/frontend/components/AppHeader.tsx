"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import DatabaseStatus from "@/components/DatabaseStatus";
import { cn } from "@/lib/utils";

const PAGES = [
	{ href: "/", label: "Chat" },
	{ href: "/ingestion", label: "Ingestion" },
	{ href: "/inspect", label: "Inspector" },
	{ href: "/doctor", label: "Doctor" },
];

export default function AppHeader() {
	const pathname = usePathname();
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
							pathname === page.href && "bg-muted text-foreground",
						)}
					>
						{page.label}
					</Link>
				))}
			</nav>
			<div className="ml-auto">
				<DatabaseStatus />
			</div>
		</header>
	);
}
