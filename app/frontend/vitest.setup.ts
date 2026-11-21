// jsdom has no ResizeObserver, which Radix measures controls with.
if (typeof window !== "undefined" && !globalThis.ResizeObserver) {
	globalThis.ResizeObserver = class {
		observe() {}
		unobserve() {}
		disconnect() {}
	};
}

// jsdom has no PointerEvent: a MouseEvent carries the coordinates canvases read.
if (typeof window !== "undefined" && !globalThis.PointerEvent) {
	globalThis.PointerEvent = class extends MouseEvent {
		readonly pointerId: number;
		constructor(type: string, init: PointerEventInit = {}) {
			super(type, init);
			this.pointerId = init.pointerId ?? 0;
		}
	} as typeof PointerEvent;
}

// jsdom has no IntersectionObserver: everything observed is in view at once.
if (typeof window !== "undefined" && !globalThis.IntersectionObserver) {
	globalThis.IntersectionObserver = class {
		constructor(private readonly callback: IntersectionObserverCallback) {}
		observe(target: Element) {
			this.callback(
				[{ isIntersecting: true, target } as IntersectionObserverEntry],
				this as unknown as IntersectionObserver,
			);
		}
		unobserve() {}
		disconnect() {}
		takeRecords() {
			return [];
		}
	} as unknown as typeof IntersectionObserver;
}
