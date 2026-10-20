import { Suspense } from "react";
import ChatPage from "@/components/chat/ChatPage";

export default function Home() {
	return (
		<Suspense>
			<ChatPage />
		</Suspense>
	);
}
