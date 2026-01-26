import { NextResponse } from "next/server";

export async function GET() {
	const backendUrl = process.env.BACKEND_URL || "http://backend:8000";

	try {
		const response = await fetch(`${backendUrl}/api/documents`);
		const data = await response.json();
		return NextResponse.json(data);
	} catch {
		return NextResponse.json(
			{ documents: [], error: "Backend unavailable" },
			{ status: 503 },
		);
	}
}
