import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	output: "standalone",
	// The compose files publish the dev server on 127.0.0.1 as well as localhost.
	allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
