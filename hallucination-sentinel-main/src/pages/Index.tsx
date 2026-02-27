import { useState, useCallback } from 'react';
import { Activity } from 'lucide-react';
import ChatPanel from '@/components/ChatPanel';
import HallucinationVerdict from '@/components/HallucinationVerdict';
import EmbeddingScatter from '@/components/EmbeddingScatter';
import EigenScoreHistogram from '@/components/EigenScoreHistogram';
import { generateMockAnalysis, getMockResponse } from '@/lib/mockData';
import type { ChatMessage, AnalysisResult } from '@/lib/mockData';

const Index = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleSend = useCallback((content: string) => {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsAnalyzing(true);

    // Simulate API delay
    setTimeout(() => {
      const response = getMockResponse();
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Generate analysis
      const result = generateMockAnalysis();
      setAnalysis(result);
      setIsAnalyzing(false);
    }, 1500);
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
          v0.1.0 — frontend only
        </span>
      </header>

      {/* Main layout */}
      <div className="flex-1 flex min-h-0">
        {/* Chat panel - left */}
        <div className="w-[380px] border-r border-border shrink-0">
          <ChatPanel messages={messages} onSend={handleSend} isAnalyzing={isAnalyzing} />
        </div>

        {/* Metrics panel - right */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {/* Verdict */}
          <HallucinationVerdict result={analysis} />

          {/* Charts grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <EmbeddingScatter result={analysis} />
            <EigenScoreHistogram scores={analysis?.eigenScores ?? null} />
          </div>

          {/* Stats row */}
          {analysis && (
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Embedding Dim', value: '768' },
                { label: 'Layers Analyzed', value: '12' },
                { label: 'Projection', value: analysis.method },
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
        </div>
      </div>
    </div>
  );
};

export default Index;
