import { useState, useCallback, useId, useRef } from 'react';
import { Activity, WifiOff, Link, Check } from 'lucide-react';
import { analyzeQuestion, getBackendUrl, setBackendUrl } from '@/lib/api';
import ChatPanel from '@/components/ChatPanel';
import HallucinationVerdict from '@/components/HallucinationVerdict';
import EmbeddingScatter from '@/components/EmbeddingScatter';
import EigenScoreHistogram from '@/components/EigenScoreHistogram';
import ResponsesPanel from '@/components/ResponsesPanel';
import type { ChatMessage, AnalysisResult } from '@/lib/mockData';

const Index = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [responses, setResponses] = useState<string[] | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useClipping, setUseClipping] = useState(true);
  const [urlInput, setUrlInput] = useState(() => getBackendUrl());
  const [savedFlash, setSavedFlash] = useState(false);
  const clippingToggleId = useId();
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleSaveUrl = useCallback((val: string) => {
    setBackendUrl(val);
    setUrlInput(getBackendUrl()); // normalise (strips trailing /)
    setSavedFlash(true);
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => setSavedFlash(false), 1500);
  }, []);

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
      const { result, responses: rawResponses, canonicalResponse, raw } = await analyzeQuestion(content, 10, useClipping);

      // Show the canonical (greedy/deterministic) response as the assistant reply
      const replyText =
        canonicalResponse?.trim() ||
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
  }, [useClipping]);

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

        {/* Feature Clipping toggle */}
        <div className="ml-auto flex items-center gap-2">
          <label
            htmlFor={clippingToggleId}
            className="text-[10px] font-mono text-muted-foreground select-none cursor-pointer"
          >
            FEAT. CLIPPING
          </label>
          {/* Pill toggle */}
          <button
            id={clippingToggleId}
            role="switch"
            aria-checked={useClipping}
            onClick={() => setUseClipping((v) => !v)}
            disabled={isAnalyzing}
            title={useClipping ? 'Feature Clipping ON — click to disable' : 'Feature Clipping OFF — click to enable'}
            className={`relative inline-flex h-4 w-8 shrink-0 cursor-pointer rounded-full border border-border transition-colors duration-200 focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-40 ${
              useClipping ? 'bg-primary' : 'bg-muted'
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-3 w-3 rounded-full bg-white shadow-sm transition-transform duration-200 mt-[1px] ${
                useClipping ? 'translate-x-4' : 'translate-x-0.5'
              }`}
            />
          </button>
        </div>

        {/* Backend URL input */}
        <div className="flex items-center gap-1.5 ml-3">
          <Link className="w-3 h-3 text-muted-foreground shrink-0" />
          <input
            id="backend-url-input"
            type="url"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onBlur={(e) => handleSaveUrl(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.currentTarget.blur(); handleSaveUrl(urlInput); } }}
            placeholder="https://xxxx.ngrok-free.app"
            aria-label="Backend URL"
            className="h-6 w-56 rounded border border-border bg-muted/40 px-2 text-[10px] font-mono text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-ring transition-colors"
          />
          {savedFlash && (
            <Check className="w-3 h-3 text-green-500 shrink-0 animate-pulse" />
          )}
        </div>

        <span className="text-[10px] font-mono text-muted-foreground">
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
