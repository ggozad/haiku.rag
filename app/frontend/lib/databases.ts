"use client";

import { useSyncExternalStore } from "react";
import { decoded } from "./http";

export interface DatabaseEntry {
	name: string;
	available: boolean;
	error: string | null;
	documents?: number;
	chunks?: number;
	supports_image_search?: boolean;
}

export interface DatabasesInfo {
	vision: boolean;
	databases: DatabaseEntry[];
}

export interface DatabasesSnapshot {
	info: DatabasesInfo | null;
	error: string | null;
}

const REFRESH_MS = 30_000;

let snapshot: DatabasesSnapshot = { info: null, error: null };
let pending: Promise<void> | null = null;
const listeners = new Set<() => void>();
let timer: ReturnType<typeof setInterval> | undefined;

function publish(next: DatabasesSnapshot) {
	snapshot = next;
	for (const listener of listeners) listener();
}

export function databasesSnapshot(): DatabasesSnapshot {
	return snapshot;
}

// Reads the configured databases once for every concurrent caller. A failure
// keeps the last list and reports the error beside it.
export function refreshDatabases(): Promise<void> {
	pending ??= fetch("/api/backend/api/databases")
		.then(async (res) => {
			publish({ info: await decoded<DatabasesInfo>(res), error: null });
		})
		.catch((error: unknown) => {
			publish({
				info: snapshot.info,
				error: error instanceof Error ? error.message : String(error),
			});
		})
		.finally(() => {
			pending = null;
		});
	return pending;
}

const whenVisible = () => {
	if (!document.hidden) void refreshDatabases();
};

// While anything subscribes, the list is read again every 30 s the page is
// visible and whenever the page becomes visible or focused.
export function subscribeDatabases(listener: () => void): () => void {
	listeners.add(listener);
	if (listeners.size === 1) {
		timer = setInterval(whenVisible, REFRESH_MS);
		document.addEventListener("visibilitychange", whenVisible);
		window.addEventListener("focus", whenVisible);
	}
	void refreshDatabases();
	return () => {
		listeners.delete(listener);
		if (listeners.size) return;
		clearInterval(timer);
		document.removeEventListener("visibilitychange", whenVisible);
		window.removeEventListener("focus", whenVisible);
	};
}

const serverSnapshot: DatabasesSnapshot = { info: null, error: null };

export function useDatabases(): DatabasesSnapshot {
	return useSyncExternalStore(
		subscribeDatabases,
		databasesSnapshot,
		() => serverSnapshot,
	);
}
