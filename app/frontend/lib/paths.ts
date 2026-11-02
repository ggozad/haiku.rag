const SEGMENTS = /^(\/[A-Za-z0-9._~-]+)+$/;

// Validates BASE_PATH: empty serves at the root, anything else is one or more
// segments of unreserved URL characters with a leading slash and no trailing
// slash.
export function basePathFrom(value: string | undefined): string {
	if (!value) return "";
	const segments = value.split("/").slice(1);
	if (
		!SEGMENTS.test(value) ||
		segments.some((segment) => segment === "." || segment === "..")
	) {
		throw new Error(
			`BASE_PATH ${JSON.stringify(value)} must be empty or a path such as "/rag": a leading slash, no trailing slash, letters, digits and "-._~" only, no "." or ".." segment.`,
		);
	}
	return value;
}

// Router navigation and <Link> carry the base path already. A URL handed to
// fetch, an <img> or a plain <a> does not.
export function withBasePath(path: string): string {
	return `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}${path}`;
}
