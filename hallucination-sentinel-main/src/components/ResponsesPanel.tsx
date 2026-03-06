/**
 * src/components/ResponsesPanel.tsx
 *
 * Shows the K raw LLM responses generated for the current question.
 * Collapsible so it doesn't crowd the metrics panel.
 */
import { useState } from 'react';
import { ChevronDown, ChevronUp, MessageSquare } from 'lucide-react';

interface ResponsesPanelProps {
    responses: string[] | null;
}

const ResponsesPanel = ({ responses }: ResponsesPanelProps) => {
    const [open, setOpen] = useState(false);

    if (!responses || responses.length === 0) {
        return (
            <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
                    Generated Responses
                </h3>
                <p className="text-sm text-muted-foreground font-mono mt-3 text-center py-4">
                    Awaiting analysis...
                </p>
            </div>
        );
    }

    return (
        <div className="rounded-lg border border-border bg-card overflow-hidden">
            {/* Header / toggle */}
            <button
                onClick={() => setOpen((o) => !o)}
                className="w-full flex items-center gap-2 px-4 py-3 hover:bg-muted/40 transition-colors text-left"
            >
                <MessageSquare className="w-4 h-4 text-primary shrink-0" />
                <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider flex-1">
                    Generated Responses
                    <span className="ml-2 text-primary">({responses.length})</span>
                </h3>
                {open ? (
                    <ChevronUp className="w-4 h-4 text-muted-foreground" />
                ) : (
                    <ChevronDown className="w-4 h-4 text-muted-foreground" />
                )}
            </button>

            {/* Collapsible body */}
            {open && (
                <div className="px-4 pb-4 space-y-2">
                    {responses.map((resp, i) => (
                        <div
                            key={i}
                            className="flex gap-3 rounded-md border border-border bg-muted/30 px-3 py-2"
                        >
                            {/* Index badge */}
                            <span className="shrink-0 w-6 h-6 flex items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-mono font-bold mt-0.5">
                                {i + 1}
                            </span>
                            <p className="text-sm font-mono text-foreground leading-relaxed break-words">
                                {resp.trim() || <span className="italic text-muted-foreground">(empty)</span>}
                            </p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ResponsesPanel;
