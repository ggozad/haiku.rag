import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
	title: "Haiku.rag Research Assistant",
	description:
		"Interactive research powered by Haiku.rag, Pydantic AI, and AG-UI",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="en">
			<body>{children}</body>
		</html>
	);
}
