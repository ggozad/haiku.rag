import type { Metadata } from "next";
import "./globals.css";
import { Geist, Geist_Mono } from "next/font/google";
import AppHeader from "@/components/AppHeader";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });
const geistMono = Geist_Mono({
	subsets: ["latin"],
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
		<html
			lang="en"
			className={cn("font-sans", geist.variable, geistMono.variable)}
		>
			<body className="flex h-dvh flex-col bg-background text-foreground antialiased">
				<TooltipProvider>
					<AppHeader />
					<main className="min-h-0 flex-1">{children}</main>
				</TooltipProvider>
			</body>
		</html>
	);
}
