"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
	type Health,
	type IngestionState,
	ingester,
	ingestionState,
} from "@/lib/ingestion";

const INTERVAL = 10000;

const VARIANTS = {
	unreachable: "destructive",
	degraded: "destructive",
	dead: "destructive",
	busy: "secondary",
	idle: "outline",
} as const;

// The ingester's state in the header, read every 10 s while the page is visible.
export default function IngestionChip() {
	const pathname = usePathname();
	const [current, setCurrent] = useState<IngestionState | null>(null);

	useEffect(() => {
		const controller = new AbortController();
		const read = async () => {
			if (document.hidden) return;
			try {
				const health = await ingester("health", { signal: controller.signal });
				setCurrent(ingestionState(health as Health));
			} catch (error) {
				if (!controller.signal.aborted)
					setCurrent(ingestionState(error as Error));
			}
		};
		void read();
		const timer = setInterval(read, INTERVAL);
		return () => {
			clearInterval(timer);
			controller.abort();
		};
	}, []);

	if (!current) return null;
	return (
		<Badge
			asChild
			variant={VARIANTS[current.state]}
			className={pathname === "/ingestion" ? "ring-2 ring-ring" : undefined}
		>
			<Link href="/ingestion">{current.label}</Link>
		</Badge>
	);
}
