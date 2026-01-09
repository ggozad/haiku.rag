import { NextResponse } from "next/server";

export async function GET(
	_request: Request,
	{ params }: { params: Promise<{ chunk_id: string }> },
) {
	const { chunk_id } = await params;
	const backendUrl = process.env.BACKEND_URL || "http://backend:8000";

	try {
		const response = await fetch(`${backendUrl}/api/visualize/${chunk_id}`);
		const data = await response.json();

		if (!response.ok) {
			return NextResponse.json(data, { status: response.status });
		}

		return NextResponse.json(data);
	} catch {
		return NextResponse.json({ error: "Backend unavailable" }, { status: 503 });
	}
}
