import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { AnalysisResult } from '@/lib/mockData';

interface EmbeddingScatterProps {
  result: AnalysisResult | null;
}

const CLUSTER_COLORS = [
  'hsl(175, 80%, 50%)',  // cyan - factual
  'hsl(0, 72%, 55%)',    // red - hallucinated
  'hsl(38, 92%, 55%)',   // amber - uncertain
];

const CLUSTER_NAMES = ['Factual', 'Hallucinated', 'Uncertain'];

const EmbeddingScatter = ({ result }: EmbeddingScatterProps) => {
  if (!result) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 h-full">
        <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          Hidden Embeddings
        </h3>
        <div className="flex items-center justify-center h-48">
          <p className="text-sm text-muted-foreground font-mono">No data yet</p>
        </div>
      </div>
    );
  }

  const clusters = [0, 1, 2].map((cluster) =>
    result.embeddings2D.filter((d) => d.cluster === cluster)
  );

  return (
    <div className="rounded-lg border border-border bg-card p-4 h-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
          Hidden Embeddings ({result.method})
        </h3>
        <span className="text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">
          {result.embeddings2D.length} points
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ top: 5, right: 5, bottom: 5, left: -15 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 14%, 18%)" />
          <XAxis
            type="number"
            dataKey="x"
            tick={{ fontSize: 10, fill: 'hsl(215, 12%, 50%)' }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(220, 14%, 18%)' }}
          />
          <YAxis
            type="number"
            dataKey="y"
            tick={{ fontSize: 10, fill: 'hsl(215, 12%, 50%)' }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(220, 14%, 18%)' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(220, 18%, 10%)',
              border: '1px solid hsl(220, 14%, 18%)',
              borderRadius: '6px',
              fontSize: '11px',
              fontFamily: 'JetBrains Mono, monospace',
              color: 'hsl(210, 20%, 90%)',
            }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Legend
            wrapperStyle={{ fontSize: '10px', fontFamily: 'JetBrains Mono, monospace' }}
          />
          {clusters.map((data, i) => (
            <Scatter
              key={i}
              name={CLUSTER_NAMES[i]}
              data={data}
              fill={CLUSTER_COLORS[i]}
              fillOpacity={0.7}
              r={4}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EmbeddingScatter;
