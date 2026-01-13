import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
	title: "haiku.rag Chat",
	description: "Conversational RAG powered by haiku.rag and AG-UI",
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
