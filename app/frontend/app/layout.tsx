import type { Metadata } from "next";
import "./globals.css";
import { Inter, JetBrains_Mono } from "next/font/google";
import AppHeader from "@/components/AppHeader";
import SearchOverlay from "@/components/SearchOverlay";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// Both carry Greek and Cyrillic glyphs.
const sans = Inter({ subsets: ["latin", "greek"], variable: "--font-sans" });
const mono = JetBrains_Mono({
	subsets: ["latin", "greek"],
	variable: "--font-mono",
	preload: false,
});

export const metadata: Metadata = {
	title: "haiku.rag",
	description: "Chat, ingestion, inspection and diagnostics for haiku.rag",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html lang="en" className={cn("font-sans", sans.variable, mono.variable)}>
			<body className="flex h-dvh flex-col bg-background text-foreground antialiased">
				<TooltipProvider>
					<AppHeader />
					<main className="min-h-0 flex-1">{children}</main>
					<SearchOverlay />
				</TooltipProvider>
			</body>
		</html>
	);
}
