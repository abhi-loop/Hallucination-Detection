import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { AnalysisResult } from '@/lib/mockData';

interface EmbeddingScatterProps {
  result: AnalysisResult | null;
}

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

  const points = result.embeddings2D;

  // Compute bounding box for axis domains with some padding
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xPad = (xMax - xMin) * 0.15 || 1;
  const yPad = (yMax - yMin) * 0.15 || 1;

  return (
    <div className="rounded-lg border border-border bg-card p-4 h-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
          Hidden Embeddings — PCA
        </h3>
        <span className="text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">
          {points.length} responses
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 5, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 14%, 18%)" />
          <XAxis
            type="number"
            dataKey="x"
            domain={[xMin - xPad, xMax + xPad]}
            tick={{ fontSize: 9, fill: 'hsl(215, 12%, 50%)' }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(220, 14%, 18%)' }}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: 'PC 1',
              position: 'insideBottom',
              offset: -2,
              fontSize: 9,
              fill: 'hsl(215, 12%, 50%)',
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[yMin - yPad, yMax + yPad]}
            tick={{ fontSize: 9, fill: 'hsl(215, 12%, 50%)' }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(220, 14%, 18%)' }}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: 'PC 2',
              angle: -90,
              position: 'insideLeft',
              dx: -12,
              fontSize: 9,
              fill: 'hsl(215, 12%, 50%)',
            }}
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
            formatter={(value: number, name: string) => [value.toFixed(4), name]}
            labelFormatter={(_: unknown, payload: { payload?: { label?: string } }[]) =>
              payload?.[0]?.payload?.label ?? ''
            }
          />
          <Scatter
            name="Response"
            data={points}
            fill="hsl(175, 80%, 50%)"
            fillOpacity={0.85}
            r={6}
          />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="text-[9px] font-mono text-muted-foreground mt-1 text-center space-y-0.5">
        <p>Tight cluster → consistent responses → lower eigenscore</p>
        <p>Spread → diverse responses → higher eigenscore</p>
      </div>
    </div>
  );
};

export default EmbeddingScatter;
