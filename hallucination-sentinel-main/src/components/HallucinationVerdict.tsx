import { Shield, ShieldAlert } from 'lucide-react';
import type { AnalysisResult } from '@/lib/mockData';

interface HallucinationVerdictProps {
  result: AnalysisResult | null;
}

const HallucinationVerdict = ({ result }: HallucinationVerdictProps) => {
  if (!result) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Verdict
        </h3>
        <div className="flex items-center justify-center py-6">
          <p className="text-sm text-muted-foreground font-mono">Awaiting analysis...</p>
        </div>
      </div>
    );
  }

  const { isHallucinated, confidence, eigenscore, threshold } = result;

  return (
    <div
      className={`rounded-lg border p-4 transition-all duration-500 ${isHallucinated
          ? 'border-destructive/40 bg-destructive/5 glow-destructive'
          : 'border-success/40 bg-success/5 glow-success'
        }`}
    >
      <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider mb-3">
        Verdict
      </h3>
      <div className="flex items-center gap-3">
        {isHallucinated ? (
          <ShieldAlert className="w-8 h-8 text-destructive" />
        ) : (
          <Shield className="w-8 h-8 text-success" />
        )}
        <div className="flex-1">
          <p
            className={`text-lg font-mono font-bold ${isHallucinated ? 'text-destructive' : 'text-success'
              }`}
          >
            {isHallucinated ? '⚠ HALLUCINATED' : '✔ FACTUAL'}
          </p>
          <p className="text-xs font-mono text-muted-foreground">
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
        </div>

        {/* Real scores badge */}
        {eigenscore !== undefined && threshold !== undefined && (
          <div className="text-right text-[10px] font-mono text-muted-foreground space-y-0.5">
            <p>
              score{' '}
              <span className="text-foreground font-bold">{eigenscore.toFixed(4)}</span>
            </p>
            <p>
              threshold{' '}
              <span className="text-foreground font-bold">{threshold.toFixed(4)}</span>
            </p>
          </div>
        )}
      </div>

      {/* Confidence bar */}
      <div className="mt-3 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${isHallucinated ? 'bg-destructive' : 'bg-success'
            }`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
    </div>
  );
};

export default HallucinationVerdict;
