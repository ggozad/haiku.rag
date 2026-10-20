import { cn } from "@/lib/utils";

// A page that scrolls as a whole: the doctor and the ingestion page.
export function Page({ children }: { children: React.ReactNode }) {
	return (
		<div className="h-full overflow-y-auto">
			<div className="mx-auto max-w-6xl space-y-6 px-4 py-6">{children}</div>
		</div>
	);
}

// The page's title, the controls and state beside it, and `aside` at the end.
export function PageHeader({
	title,
	children,
	aside,
}: {
	title: string;
	children?: React.ReactNode;
	aside?: React.ReactNode;
}) {
	return (
		<header className="flex flex-wrap items-center gap-3">
			<h1 className="text-lg font-semibold">{title}</h1>
			{children}
			{aside && <div className="ml-auto">{aside}</div>}
		</header>
	);
}

export function SectionHeading({
	children,
	className,
}: {
	children: React.ReactNode;
	className?: string;
}) {
	return <h2 className={cn("text-sm font-semibold", className)}>{children}</h2>;
}
