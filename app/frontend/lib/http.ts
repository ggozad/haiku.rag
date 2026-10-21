// The JSON body of a response, or an Error naming why there is none: the
// `error` or `detail` a failure carries, else its status.
export async function decoded<T = unknown>(response: Response): Promise<T> {
	const text = await response.text();
	let data: unknown;
	try {
		data = JSON.parse(text);
	} catch {
		if (!response.ok)
			throw new Error(`${response.status} ${response.statusText}`.trim());
		throw new Error("the response is not JSON");
	}
	if (!response.ok) {
		const body = data as { error?: unknown; detail?: unknown } | null;
		throw new Error(
			String(body?.error ?? body?.detail ?? "") ||
				`${response.status} ${response.statusText}`.trim(),
		);
	}
	return data as T;
}
