import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface EigenScoreHistogramProps {
  scores: number[] | null;
  liveScore?: number;        // current question's eigenscore
  isHallucinated?: boolean;  // used to colour the live marker
}

const EigenScoreHistogram = ({ scores, liveScore, isHallucinated }: EigenScoreHistogramProps) => {
  if (!scores) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 h-full">
        <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          EigenScore Distribution
        </h3>
        <div className="flex items-center justify-center h-48">
          <p className="text-sm text-muted-foreground font-mono">No data yet</p>
        </div>
      </div>
    );
  }

  // Dynamic binning that works with any real-valued range (e.g. -2 to +3)
  const numBins = 12;
  const minVal = Math.min(...scores);
  const maxVal = Math.max(...scores);
  const range = maxVal - minVal || 1; // avoid div-by-zero

  const bins = Array.from({ length: numBins }, (_, i) => {
    const binStart = minVal + (i / numBins) * range;
    const binEnd = minVal + ((i + 1) / numBins) * range;
    return { range: binStart.toFixed(2), count: 0, binStart, binEnd };
  });

  scores.forEach((s) => {
    const idx = Math.min(
      Math.floor(((s - minVal) / range) * numBins),
      numBins - 1
    );
    if (idx >= 0 && idx < numBins) bins[idx].count++;
  });

  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const meanBinIdx = Math.min(
    Math.max(Math.floor(((mean - minVal) / range) * numBins), 0),
    numBins - 1
  );

  // Live score marker — clamped to the reference range
  const liveBinIdx =
    liveScore !== undefined
      ? Math.min(
        Math.max(Math.floor(((liveScore - minVal) / range) * numBins), 0),
        numBins - 1
      )
      : null;
  const liveColor = isHallucinated ? 'hsl(0, 72%, 55%)' : 'hsl(145, 70%, 45%)';

  return (
    <div className="rounded-lg border border-border bg-card p-4 h-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
          EigenScore Distribution
        </h3>
        <span className="text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">
          μ = {mean.toFixed(3)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={bins} margin={{ top: 5, right: 5, bottom: 5, left: -15 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 14%, 18%)" vertical={false} />
          <XAxis
            dataKey="range"
            tick={{ fontSize: 9, fill: 'hsl(215, 12%, 50%)' }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(220, 14%, 18%)' }}
            interval={1}
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'hsl(215, 12%, 50%)' }}
            tickLine={false}
            axisLine={{ stroke: 'hsl(220, 14%, 18%)' }}
            allowDecimals={false}
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
            formatter={(value: number) => [`${value} samples`, 'Count']}
            labelFormatter={(label) => `Score: ${label}`}
          />
          <ReferenceLine
            x={bins[meanBinIdx].range}
            stroke="hsl(175, 80%, 50%)"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{
              value: 'mean',
              position: 'top',
              fill: 'hsl(175, 80%, 50%)',
              fontSize: 10,
              fontFamily: 'JetBrains Mono, monospace',
            }}
          />
          {liveBinIdx !== null && (
            <ReferenceLine
              x={bins[liveBinIdx].range}
              stroke={liveColor}
              strokeWidth={2}
              label={{
                value: `live ${liveScore!.toFixed(2)}`,
                position: 'top',
                fill: liveColor,
                fontSize: 10,
                fontFamily: 'JetBrains Mono, monospace',
              }}
            />
          )}
          <Bar
            dataKey="count"
            fill="hsl(215, 80%, 55%)"
            fillOpacity={0.8}
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EigenScoreHistogram;
