"use client";

import { MessageSquare } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { askAbout } from "@/lib/sessions";

// Opens the chat on a new session that searches only this document.
export default function AskAbout({
	source,
	documentId,
}: {
	source: string;
	documentId: string;
}) {
	const router = useRouter();
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState<string | null>(null);
	return (
		<span className="inline-flex flex-wrap items-center gap-2">
			<Button
				size="sm"
				variant="outline"
				disabled={busy}
				onClick={async () => {
					setBusy(true);
					setError(null);
					try {
						const id = await askAbout({ source, id: documentId });
						router.push(`/?session=${encodeURIComponent(id)}`);
					} catch (err) {
						setError(err instanceof Error ? err.message : String(err));
						setBusy(false);
					}
				}}
			>
				<MessageSquare />
				Ask about this document
			</Button>
			{error && <span className="text-sm text-destructive">{error}</span>}
		</span>
	);
}
