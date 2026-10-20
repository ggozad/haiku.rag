const BASE = "/api/ingester";

// The ingester dashboard's cycle: what one refresh reads.
export const CYCLE = {
	health: "health",
	sources: "sources",
	providers: "providers",
	stats: "stats",
	active: "jobs?status=claimed&limit=20",
	dead: "jobs?status=dead&limit=10",
	succeeded: "jobs?status=succeeded&limit=10",
} as const;

export type Panel = keyof typeof CYCLE;

export interface Reading {
	data?: unknown;
	error?: string;
}

export type Snapshot = Record<Panel, Reading> & {
	// When this cycle's answers arrived.
	at: number;
	// When a cycle last found the ingester answering its health call.
	healthyAt: number | null;
};

export function describeFailure(status: number, detail: string): string {
	if (status === 401) {
		return "The ingester refused the request: set INGESTER_TOKEN in .env to its api.auth_token.";
	}
	return detail || `The ingester answered ${status}.`;
}

// Gives up after `timeoutMs`, or when the caller's signal aborts.
export async function ingester(
	path: string,
	init: RequestInit = {},
	timeoutMs = 10000,
): Promise<unknown> {
	const controller = new AbortController();
	let late = false;
	const timeout = setTimeout(() => {
		late = true;
		controller.abort();
	}, timeoutMs);
	const leave = () => controller.abort();
	if (init.signal?.aborted) controller.abort();
	init.signal?.addEventListener("abort", leave);
	let response: Response;
	let text: string;
	try {
		response = await fetch(`${BASE}/${path}`, {
			...init,
			signal: controller.signal,
		});
		text = await response.text();
	} catch (error) {
		if (late) throw new Error("The ingester did not answer in time.");
		throw error;
	} finally {
		clearTimeout(timeout);
		init.signal?.removeEventListener("abort", leave);
	}
	let data: unknown = null;
	try {
		data = text ? JSON.parse(text) : null;
	} catch {
		data = text;
	}
	if (!response.ok) {
		const detail =
			(data as { detail?: unknown; error?: unknown } | null)?.detail ??
			(data as { error?: unknown } | null)?.error;
		throw new Error(describeFailure(response.status, String(detail ?? "")));
	}
	return data;
}

// Runs the cycle every `intervalMs`. One cycle runs at a time: a tick that finds
// one in flight is skipped. Each cycle carries a sequence number and is aborted
// after `timeoutMs`; a cycle superseded by `refresh` or `stop` is discarded.
export class Poller {
	private sequence = 0;
	private inFlight: AbortController | null = null;
	private timer: ReturnType<typeof setInterval> | null = null;
	healthyAt: number | null = null;

	constructor(
		private readonly onSnapshot: (snapshot: Snapshot) => void,
		private readonly intervalMs = 3000,
		private readonly timeoutMs = 10000,
	) {}

	start(): void {
		if (this.timer !== null) return;
		this.timer = setInterval(() => {
			void this.cycle();
		}, this.intervalMs);
		void this.cycle();
	}

	stop(): void {
		if (this.timer !== null) clearInterval(this.timer);
		this.timer = null;
		this.interrupt();
	}

	// After an action: drop the cycle in flight and read again now, unless
	// polling stopped meanwhile.
	refresh(): Promise<void> {
		this.interrupt();
		return this.timer === null ? Promise.resolve() : this.cycle();
	}

	// Drops the cycle in flight, as an action starts.
	interrupt(): void {
		this.sequence++;
		this.inFlight?.abort();
		this.inFlight = null;
	}

	async cycle(): Promise<void> {
		if (this.inFlight) return;
		const mine = ++this.sequence;
		const controller = new AbortController();
		this.inFlight = controller;
		const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
		const entries = await Promise.all(
			(Object.entries(CYCLE) as [Panel, string][]).map(
				async ([panel, path]): Promise<[Panel, Reading]> => {
					try {
						return [
							panel,
							{ data: await ingester(path, { signal: controller.signal }) },
						];
					} catch (error) {
						const message = controller.signal.aborted
							? "The ingester did not answer in time."
							: error instanceof Error
								? error.message
								: String(error);
						return [panel, { error: message }];
					}
				},
			),
		);
		clearTimeout(timeout);
		if (mine !== this.sequence) return;
		this.inFlight = null;
		const readings = Object.fromEntries(entries) as Record<Panel, Reading>;
		const at = Date.now();
		if (!readings.health.error) this.healthyAt = at;
		this.onSnapshot({ ...readings, at, healthyAt: this.healthyAt });
	}
}

export function formatAge(seconds: number | null | undefined): string {
	if (seconds === null || seconds === undefined) return "—";
	if (seconds < 60) return `${Math.round(seconds)}s`;
	if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
	return `${(seconds / 3600).toFixed(1)}h`;
}

export interface IngestionState {
	state: "unreachable" | "degraded" | "dead" | "busy" | "idle";
	label: string;
}

export interface Health {
	status: string;
	queue_counts: Record<string, number>;
}

// The one state the header shows: the first that holds, in this order.
export function ingestionState(health: Health | Error): IngestionState {
	if (health instanceof Error)
		return { state: "unreachable", label: "Ingester unreachable" };
	if (health.status !== "ok")
		return { state: "degraded", label: "Ingester degraded" };
	const counts = health.queue_counts;
	const dead = counts.dead ?? 0;
	if (dead)
		return {
			state: "dead",
			label: `${dead} dead ${dead === 1 ? "letter" : "letters"}`,
		};
	const pending = (counts.queued ?? 0) + (counts.claimed ?? 0);
	if (pending) return { state: "busy", label: `Ingesting ${pending}` };
	return { state: "idle", label: "Ingester idle" };
}
