import { Suspense } from "react";
import InspectPage from "@/components/inspect/InspectPage";

// The inspector, `/inspect/…` and `/search`, reads its view from the address
// and stays mounted while the address changes, so its pages render nothing.
export default function InspectLayout() {
	return (
		<Suspense>
			<InspectPage />
		</Suspense>
	);
}
