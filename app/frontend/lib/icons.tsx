interface IconProps {
	size?: number;
	strokeWidth?: number;
}

export function BrainIcon({ size = 18, strokeWidth = 2 }: IconProps) {
	return (
		<svg
			width={size}
			height={size}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth={strokeWidth}
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
			<path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
			<path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
			<path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
			<path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
			<path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
			<path d="M19.938 10.5a4 4 0 0 1 .585.396" />
			<path d="M6 18a4 4 0 0 1-1.967-.516" />
			<path d="M19.967 17.484A4 4 0 0 1 18 18" />
		</svg>
	);
}

export function FilterIcon({ size = 18, strokeWidth = 2 }: IconProps) {
	return (
		<svg
			width={size}
			height={size}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth={strokeWidth}
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
		</svg>
	);
}
