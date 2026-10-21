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
