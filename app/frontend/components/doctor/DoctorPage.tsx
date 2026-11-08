"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect } from "react";
import { Page, PageHeader, SectionHeading } from "@/components/Page";
import { Empty } from "@/components/Status";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { doctorHref, documentHref } from "@/lib/addresses";
import { useDatabases } from "@/lib/databases";
import ChecksPanel from "./ChecksPanel";
import { Estimate, useEstimate } from "./Estimate";
import {
	FamilyList,
	SimilarityControls,
	useSimilarityJob,
} from "./SimilarityJob";

function Similarity({ source, ready }: { source: string; ready: boolean }) {
	const similarity = useSimilarityJob(source);
	return (
		<section className="space-y-3">
			<SectionHeading>Document similarity</SectionHeading>
			<SimilarityControls similarity={similarity} ready={ready} />
			{similarity.result && (
				<div className="max-w-xl text-sm">
					<FamilyList
						source={source}
						families={similarity.result.families}
						hrefOf={(family) => documentHref(source, family.keep)}
					/>
				</div>
			)}
		</section>
	);
}

function DoctorDatabase({ source }: { source: string }) {
	const estimate = useEstimate(source);
	return (
		<>
			<ChecksPanel source={source} ready={estimate.ready} />
			<Similarity source={source} ready={estimate.ready} />
			<div className="space-y-1 border-t pt-4">
				<Estimate text={estimate.text} />
				<p className="text-xs text-muted-foreground">
					Results are kept until the next run or a restart of the backend, and
					one job runs at a time across databases.
				</p>
			</div>
		</>
	);
}

export default function DoctorPage() {
	const { info } = useDatabases();
	const databases = info?.databases ?? [];
	const router = useRouter();
	const named = useParams<{ db?: string[] }>().db?.[0] ?? null;
	const first = databases[0]?.name;
	useEffect(() => {
		if (named === null && first) router.replace(doctorHref(first));
	}, [named, first, router]);
	const known =
		info && named !== null && databases.some((db) => db.name === named);
	const source = known ? named : null;

	return (
		<Page>
			<PageHeader title="Doctor">
				<Select
					value={source ?? ""}
					onValueChange={(name) => router.push(doctorHref(name))}
				>
					<SelectTrigger className="w-56" aria-label="Database">
						<SelectValue placeholder="Database" />
					</SelectTrigger>
					<SelectContent>
						{databases.map((db) => (
							<SelectItem key={db.name} value={db.name}>
								{db.name}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
			</PageHeader>
			{info && named !== null && !known && (
				<Empty>No database is named {named}.</Empty>
			)}
			{source && <DoctorDatabase key={source} source={source} />}
		</Page>
	);
}
