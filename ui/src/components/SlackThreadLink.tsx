interface Props {
  href: string;
  className?: string;
  label?: string;
}

// "Open in Slack" affordance. Render ONLY when slackPermalink(...) !== null.
// Opens the shared thread in a new tab; stopPropagation prevents a parent
// row/card onSelect from also firing.
export default function SlackThreadLink({ href, className, label = "Open Slack thread" }: Props) {
  return (
    <a
      className={`slack-thread-link ${className ?? ""}`.trim()}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      title={label}
      aria-label={label}
      onClick={(e) => e.stopPropagation()}
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M10 14 21 3M15 3h6v6M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
      </svg>
    </a>
  );
}
