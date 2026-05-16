// Mock data generators for the hallucination detection dashboard

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface AnalysisResult {
  isHallucinated: boolean;
  confidence: number;
  eigenScores: number[];
  embeddings2D: { x: number; y: number; label: string; cluster: number }[];
  method: 'PCA' | 't-SNE';
  // Real backend fields (undefined when using mock data)
  eigenscore?: number;
  threshold?: number;
  gMean?: number;
}

// Generate mock PCA/t-SNE embeddings
export function generateMockEmbeddings(count: number = 80): AnalysisResult['embeddings2D'] {
  const clusters = [
    { cx: -2, cy: 1, label: 'Factual', cluster: 0 },
    { cx: 2.5, cy: -1.5, label: 'Hallucinated', cluster: 1 },
    { cx: -1, cy: -2, label: 'Uncertain', cluster: 2 },
  ];

  return Array.from({ length: count }, (_, i) => {
    const c = clusters[i % 3];
    return {
      x: c.cx + (Math.random() - 0.5) * 2.5,
      y: c.cy + (Math.random() - 0.5) * 2.5,
      label: c.label,
      cluster: c.cluster,
    };
  });
}

// Generate mock EigenScores
export function generateMockEigenScores(count: number = 30): number[] {
  return Array.from({ length: count }, () =>
    Math.max(0, Math.min(1, 0.3 + Math.random() * 0.5 + (Math.random() > 0.7 ? 0.3 : 0)))
  );
}

// Generate a mock analysis result
export function generateMockAnalysis(): AnalysisResult {
  const isHallucinated = Math.random() > 0.5;
  return {
    isHallucinated,
    confidence: 0.7 + Math.random() * 0.25,
    eigenScores: generateMockEigenScores(),
    embeddings2D: generateMockEmbeddings(),
    method: Math.random() > 0.5 ? 'PCA' : 't-SNE',
  };
}

// Mock responses
const mockResponses = [
  "The Eiffel Tower was built in 1889 as the entrance arch for the World's Fair. It stands at 330 meters tall and was designed by Gustave Eiffel's engineering company.",
  "Water boils at 100°C (212°F) at standard atmospheric pressure. This is a fundamental physical property used in cooking and industrial processes.",
  "The Great Wall of China is visible from space with the naked eye, stretching over 13,000 miles across northern China.",
  "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy, primarily occurring in chloroplasts of plant cells.",
  "Albert Einstein invented the lightbulb in 1879 while working at the Swiss Patent Office in Bern.",
];

export function getMockResponse(): string {
  return mockResponses[Math.floor(Math.random() * mockResponses.length)];
}
