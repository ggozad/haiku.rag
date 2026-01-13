import { NextResponse } from "next/server";

export async function GET() {
	const backendUrl = process.env.BACKEND_URL || "http://backend:8000";

	try {
		const response = await fetch(`${backendUrl}/api/info`);
		const data = await response.json();
		return NextResponse.json(data);
	} catch {
		return NextResponse.json(
			{ exists: false, error: "Backend unavailable" },
			{ status: 503 },
		);
	}
}
