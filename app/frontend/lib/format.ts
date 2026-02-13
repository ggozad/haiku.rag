export function formatRelativeTime(dateStr: string, compact = false): string {
	const now = Date.now();
	const then = new Date(dateStr).getTime();
	const seconds = Math.floor((now - then) / 1000);
	if (seconds < 60) return "just now";
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60)
		return compact
			? `${minutes}m ago`
			: `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
	const hours = Math.floor(minutes / 60);
	if (hours < 24)
		return compact
			? `${hours}h ago`
			: `${hours} hour${hours === 1 ? "" : "s"} ago`;
	const days = Math.floor(hours / 24);
	return compact ? `${days}d ago` : new Date(dateStr).toLocaleDateString();
}
