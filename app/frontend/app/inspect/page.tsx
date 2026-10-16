import { Suspense } from "react";
import InspectPage from "@/components/inspect/InspectPage";

export default function Inspect() {
	return (
		<Suspense>
			<InspectPage />
		</Suspense>
	);
}
