import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function Markdown({ children }: { children: string }) {
	return (
		<div className="space-y-3 text-sm leading-relaxed [&_a]:text-primary [&_a]:underline [&_blockquote]:border-l-2 [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:font-mono [&_code]:text-[0.85em] [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:text-base [&_h2]:font-semibold [&_h3]:font-semibold [&_li]:my-0.5 [&_ol]:list-decimal [&_ol]:pl-5 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-muted [&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_table]:w-full [&_table]:text-left [&_td]:border-t [&_td]:px-2 [&_td]:py-1 [&_th]:px-2 [&_th]:py-1 [&_th]:font-medium [&_ul]:list-disc [&_ul]:pl-5">
			<ReactMarkdown
				remarkPlugins={[remarkGfm]}
				components={{
					a: ({ node: _node, ...props }) => (
						<a {...props} target="_blank" rel="noreferrer" />
					),
					table: ({ node: _node, ...props }) => (
						<div className="overflow-x-auto">
							<table {...props} />
						</div>
					),
				}}
			>
				{children}
			</ReactMarkdown>
		</div>
	);
}
