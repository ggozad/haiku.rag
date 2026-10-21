// What a view shows while it loads, or when it has nothing to show.

export function Loading() {
	return <p className="text-sm text-muted-foreground">Loading…</p>;
}

// Fills the panel it stands in, centred.
export function Empty({ children }: { children: React.ReactNode }) {
	return (
		<div className="flex min-h-40 flex-1 items-center justify-center p-6 text-center text-sm text-muted-foreground">
			{children}
		</div>
	);
}
