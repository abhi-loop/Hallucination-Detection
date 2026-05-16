/**
 * src/lib/api.ts
 *
 * Typed fetch wrapper for the FastAPI /analyze endpoint.
 * The backend URL is read from (in priority order):
 *   1. localStorage key "backendUrl"  — set via the UI URL field
 *   2. VITE_API_URL env variable       — set at build time for Vercel
 *   3. hardcoded fallback
 */

import type { AnalysisResult } from './mockData';

const LS_KEY = 'backendUrl';
const ENV_DEFAULT = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/** Returns the currently active backend URL. */
export function getBackendUrl(): string {
  return localStorage.getItem(LS_KEY) ?? ENV_DEFAULT;
}

/** Persists a new backend URL to localStorage so it survives page reloads. */
export function setBackendUrl(url: string): void {
  const trimmed = url.trim().replace(/\/+$/, ''); // strip trailing slashes
  if (trimmed) {
    localStorage.setItem(LS_KEY, trimmed);
  } else {
    localStorage.removeItem(LS_KEY);
  }
}

// ── Shape returned by the Python server ───────────────────────────────────────

export interface AnalyzeResponse {
    eigenscore: number;
    threshold: number;
    g_mean: number;
    verdict: 'factual' | 'hallucination';
    confidence: number;
    canonical_response: string;        // greedy deterministic answer for chat display
    responses: string[];               // all K generated answers (for EigenScore)
    eigen_scores_ref: number[];        // reference scores from results.csv
    embeddings_2d: {
        x: number;
        y: number;
        label: string;
        cluster: number;
    }[];
}

// ── Health check ─────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
    try {
        const res = await fetch(`${getBackendUrl()}/health`, { method: 'GET' });
        if (!res.ok) return false;
        const data = await res.json();
        return data?.model_loaded === true;
    } catch {
        return false;
    }
}

// ── Main inference call ───────────────────────────────────────────────────────

export async function analyzeQuestion(
    question: string,
    k: number = 10,
    useClipping: boolean = true
): Promise<{ raw: AnalyzeResponse; result: AnalysisResult; responses: string[]; canonicalResponse: string }> {
    // 5-minute timeout — LLM inference can take ~30-60 s
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5 * 60 * 1000);

    let res: Response;
    try {
        res = await fetch(`${getBackendUrl()}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, k, use_clipping: useClipping }),
            signal: controller.signal,
        });
    } catch (err) {
        clearTimeout(timer);
        if (err instanceof DOMException && err.name === 'AbortError') {
            throw new Error('Request timed out after 5 minutes.');
        }
        throw new Error(
            'Cannot reach backend. Make sure the server is running:\n' +
            'env\\scripts\\uvicorn api.server:app --port 8000'
        );
    }
    clearTimeout(timer);

    if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new Error(`Backend error ${res.status}: ${detail}`);
    }

    const raw: AnalyzeResponse = await res.json();

    // Map to the AnalysisResult shape the existing components expect
    const result: AnalysisResult = {
        isHallucinated: raw.verdict === 'hallucination',
        confidence: raw.confidence,
        eigenScores: raw.eigen_scores_ref,
        embeddings2D: raw.embeddings_2d,
        method: 'PCA',
        // Extra real data passed through
        eigenscore: raw.eigenscore,
        threshold: raw.threshold,
        gMean: raw.g_mean,
    };

    return { raw, result, responses: raw.responses, canonicalResponse: raw.canonical_response };
}
