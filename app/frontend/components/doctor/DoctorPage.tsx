"use client";

import Link from "next/link";
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
import { databaseHref, doctorHref } from "@/lib/addresses";
import { useDatabases } from "@/lib/databases";
import ChecksPanel from "./ChecksPanel";
import {
	Estimate,
	FamilyList,
	MapControls,
	useEstimate,
	useMapJob,
} from "./MapJob";

const mapHref = (source: string, doc?: string) =>
	databaseHref(
		source,
		new URLSearchParams(doc ? { view: "map", doc } : { view: "map" }),
	);

function CorpusMapJob({ source, ready }: { source: string; ready: boolean }) {
	const map = useMapJob(source);
	return (
		<section className="space-y-3">
			<div className="flex flex-wrap items-center gap-3">
				<SectionHeading>Corpus map</SectionHeading>
				<Link href={mapHref(source)} className="text-sm underline">
					Open the map
				</Link>
			</div>
			<MapControls map={map} ready={ready} />
			{map.result && (
				<div className="max-w-xl text-sm">
					<FamilyList
						source={source}
						families={map.result.families}
						hrefOf={(family) => mapHref(source, family.keep)}
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
			<Estimate text={estimate.text} />
			<p className="text-xs text-muted-foreground">
				Results are kept until the next run or a restart of the backend, and one
				job runs at a time across databases.
			</p>
			<CorpusMapJob source={source} ready={estimate.ready} />
			<ChecksPanel source={source} ready={estimate.ready} />
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
