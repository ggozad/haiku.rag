export interface Health {
	status: string;
	queue_counts: Record<string, number>;
	worker_count: number;
	poller_count: number;
	workers_alive: number;
	pollers_alive: number;
	worker_breaker_open: boolean;
	worker_breaker_consecutive_failures: number;
}

export interface Stats {
	throughput: {
		succeeded_5m: number;
		succeeded_30m: number;
		succeeded_1h: number;
	};
	workers: { busy: number; total: number };
	oldest_queued_age_s: number | null;
	dlq_by_source: Record<string, number>;
	queue_depth_by_source: Record<string, number>;
}

export interface Source {
	source_id: string;
	type: string;
	last_polled_at: string | null;
	circuit_breaker_open: boolean;
	last_skip_reason: string | null;
}

export interface Providers {
	docling_serve: {
		base_url: string;
		reachable: boolean;
		status_code: number | null;
		error: string | null;
	}[];
}

export interface Job {
	id: string;
	source_id: string;
	uri: string;
	op: string;
	content_hash: string | null;
	revision: string | null;
	status: string;
	attempts: number;
	max_attempts: number;
	last_error: string | null;
	conversion_stalled: boolean;
	extra: Record<string, unknown> | null;
	enqueued_at: string;
	scheduled_at: string;
	claimed_at: string | null;
	claimed_by: string | null;
	last_heartbeat_at: string | null;
	completed_at: string | null;
}
