import { Highlight } from "prism-react-renderer";

// Colours come from the .token rules in globals.css, which follow the colour scheme.
const UNSTYLED = { plain: {}, styles: [] };

export default function Highlighted({
	code,
	language,
	className,
}: {
	code: string;
	language: "python" | "yaml";
	className: string;
}) {
	return (
		<Highlight code={code} language={language} theme={UNSTYLED}>
			{({ tokens, getLineProps, getTokenProps }) => (
				<pre className={className}>
					{tokens.map((line, n) => (
						// biome-ignore lint/suspicious/noArrayIndexKey: lines have no identity of their own
						<div key={n} {...getLineProps({ line })}>
							{line.map((token, i) => (
								// biome-ignore lint/suspicious/noArrayIndexKey: tokens have no identity of their own
								<span key={i} {...getTokenProps({ token })} />
							))}
						</div>
					))}
				</pre>
			)}
		</Highlight>
	);
}
