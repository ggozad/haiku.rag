export function readableUri(uri: string): string {
	try {
		return decodeURI(uri);
	} catch {
		return uri;
	}
}

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

// The document's web address, when its metadata names an http(s) one.
export function publicUrl(meta?: Record<string, unknown>): string | null {
	const url = meta?.url;
	return typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
}

export function formatBytes(n: number): string {
	if (n < 1024) return `${n} B`;
	const units = ["KB", "MB", "GB", "TB"];
	let value = n / 1024;
	let unit = 0;
	while (value >= 1024 && unit < units.length - 1) {
		value /= 1024;
		unit++;
	}
	return `${value.toFixed(1)} ${units[unit]}`;
}

// `HH:MM:SS-HH:MM:SS`, or one time when both ends fall in the same second.
export function formatTimeRange(
	range: [number, number] | null | undefined,
): string | null {
	if (!range) return null;
	const clock = (seconds: number) => {
		const whole = Math.floor(seconds);
		return [whole / 3600, (whole % 3600) / 60, whole % 60]
			.map((n) => String(Math.floor(n)).padStart(2, "0"))
			.join(":");
	};
	const [start, end] = range.map(clock);
	return start === end ? start : `${start}-${end}`;
}
