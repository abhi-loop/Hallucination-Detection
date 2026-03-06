import { useState, useCallback } from 'react';
import { Activity, WifiOff } from 'lucide-react';
import ChatPanel from '@/components/ChatPanel';
import HallucinationVerdict from '@/components/HallucinationVerdict';
import EmbeddingScatter from '@/components/EmbeddingScatter';
import EigenScoreHistogram from '@/components/EigenScoreHistogram';
import ResponsesPanel from '@/components/ResponsesPanel';
import { analyzeQuestion } from '@/lib/api';
import type { ChatMessage, AnalysisResult } from '@/lib/mockData';

const Index = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [responses, setResponses] = useState<string[] | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = useCallback(async (content: string) => {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsAnalyzing(true);
    setError(null);

    try {
      const { result, responses: rawResponses, raw } = await analyzeQuestion(content);

      // Show the first (or best) response as the assistant reply
      const replyText =
        rawResponses[0]?.trim() ||
        `[EigenScore: ${raw.eigenscore.toFixed(4)} | Verdict: ${raw.verdict.toUpperCase()}]`;

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: replyText,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setAnalysis(result);
      setResponses(rawResponses);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);

      const errMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `⚠ Backend error: ${msg}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleClear = useCallback(() => {
    setMessages([]);
    setAnalysis(null);
    setResponses(null);
    setError(null);
  }, []);

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Top bar */}
      <header className="border-b border-border px-4 py-2.5 flex items-center gap-3 shrink-0">
        <Activity className="w-5 h-5 text-primary" />
        <h1 className="text-sm font-mono font-bold text-foreground tracking-wide">
          HALLUCINATION DETECTOR
        </h1>
        <span className="text-[10px] font-mono text-muted-foreground ml-auto">
          v1.0.0 — live backend
        </span>
      </header>

      {/* Backend error banner */}
      {error && (
        <div className="flex items-center gap-2 px-4 py-2 bg-destructive/10 border-b border-destructive/20 text-destructive text-xs font-mono">
          <WifiOff className="w-3 h-3 shrink-0" />
          <span>
            {error.includes('fetch') || error.includes('network')
              ? 'Cannot reach backend. Make sure the FastAPI server is running on port 8000.'
              : error}
          </span>
        </div>
      )}

      {/* Main layout */}
      <div className="flex-1 flex min-h-0">
        {/* Chat panel - left */}
        <div className="w-[380px] border-r border-border shrink-0">
          <ChatPanel messages={messages} onSend={handleSend} onClear={handleClear} isAnalyzing={isAnalyzing} />
        </div>

        {/* Metrics panel - right */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {/* Verdict */}
          <HallucinationVerdict result={analysis} />

          {/* Charts grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <EmbeddingScatter result={analysis} />
            <EigenScoreHistogram
              scores={analysis?.eigenScores ?? null}
              liveScore={analysis?.eigenscore}
              isHallucinated={analysis?.isHallucinated}
            />
          </div>

          {/* Stats row */}
          {analysis && (
            <div className="grid grid-cols-3 gap-3">
              {[
                {
                  label: 'EigenScore',
                  value:
                    analysis.eigenscore !== undefined
                      ? analysis.eigenscore.toFixed(4)
                      : 'N/A',
                },
                {
                  label: 'Threshold',
                  value:
                    analysis.threshold !== undefined
                      ? analysis.threshold.toFixed(4)
                      : 'N/A',
                },
                {
                  label: 'G-Mean',
                  value:
                    analysis.gMean !== undefined
                      ? (analysis.gMean * 100).toFixed(1) + '%'
                      : analysis.method,
                },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-lg border border-border bg-card p-3 text-center"
                >
                  <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">
                    {stat.label}
                  </p>
                  <p className="text-lg font-mono font-bold text-foreground mt-1">
                    {stat.value}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* K Responses panel */}
          <ResponsesPanel responses={responses} />
        </div>
      </div>
    </div>
  );
};

export default Index;
