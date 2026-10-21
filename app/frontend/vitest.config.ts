import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
	resolve: {
		alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
	},
	esbuild: { jsx: "automatic" },
	// Component tests choose jsdom with an `@vitest-environment` comment.
	test: {
		include: ["lib/**/*.test.ts", "components/**/*.test.tsx"],
		setupFiles: ["./vitest.setup.ts"],
	},
});
