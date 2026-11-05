import type { NextConfig } from "next";
import { basePathFrom } from "./lib/paths";

const basePath = basePathFrom(process.env.BASE_PATH);

const nextConfig: NextConfig = {
	output: "standalone",
	basePath,
	env: { NEXT_PUBLIC_BASE_PATH: basePath },
	// The compose files publish the dev server on 127.0.0.1 as well as localhost.
	allowedDevOrigins: ["127.0.0.1"],
	// The dev server's route indicator covers the inspector's pagination.
	devIndicators: false,
};

export default nextConfig;
