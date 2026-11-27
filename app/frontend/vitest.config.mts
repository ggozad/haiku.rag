import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
	resolve: {
		alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
	},
	// Component tests choose jsdom with an `@vitest-environment` comment.
	test: {
		include: ["lib/**/*.test.ts", "components/**/*.test.tsx"],
		setupFiles: ["./vitest.setup.ts"],
		// Under node, pdf.js runs its worker in-process. A loading task destroyed
		// before its document loads rejects inside the worker's own load chain,
		// which in a browser is the worker thread.
		onUnhandledError: (error) =>
			!(
				error.message === "Worker was terminated" &&
				String(error.stack).includes("pdf.worker")
			),
	},
});
