'use client'

import { useState } from 'react'
import { TetrisBoard } from '@/components/tetris/tetris-board'
import { ScoreChart } from '@/components/tetris/score-chart'
import { GenerationList } from '@/components/tetris/generation-list'
import { ControlPanel } from '@/components/tetris/control-panel'
import { StatsPanel } from '@/components/tetris/stats-panel'
import { NextPiece } from '@/components/tetris/next-piece'
import { TrainingStatus } from '@/components/tetris/training-status'
import { Brain, Sparkles } from 'lucide-react'

const MOCK_BOARD: (string | null)[][] = (() => {
  const board: (string | null)[][] = Array(20).fill(null).map(() => Array(10).fill(null))
  const pattern: [number, number, string][] = [
    [19,0,'J'],[19,1,'J'],[19,2,'J'],[19,3,'T'],[19,4,'T'],[19,5,'T'],[19,6,'O'],[19,7,'O'],[19,8,'L'],[19,9,'L'],
    [18,0,'J'],[18,2,'S'],[18,3,'S'],[18,4,'T'],[18,6,'O'],[18,7,'O'],[18,8,'L'],
    [17,2,'S'],[17,3,'S'],[17,5,'Z'],[17,6,'Z'],[17,8,'I'],[17,9,'I'],
    [16,3,'I'],[16,4,'I'],[16,5,'I'],[16,6,'I'],
    [15,1,'T'],[15,2,'T'],[15,3,'T'],[15,7,'J'],[15,8,'J'],[15,9,'J'],
    [14,2,'T'],[14,7,'J'],
  ]
  for (const [r, c, p] of pattern) {
    board[r][c] = p
  }
  return board
})()

const MOCK_ACTIVE_PIECE = {
  type: 'T',
  position: { row: 5, col: 4 },
  shape: [[0, 1, 0], [1, 1, 1]],
  rotation: 0,
}

const MOCK_CHART_DATA = Array.from({ length: 42 }, (_, i) => ({
  generation: i + 1,
  score: Math.floor(1000 + (i * 200) + Math.sin(i * 0.5) * 500),
  avgScore: Math.floor(800 + (i * 150) + Math.cos(i * 0.3) * 300),
}))

const MOCK_GENERATIONS = [
  { id: 42, score: 12840, linesCleared: 128, timestamp: '2 min ago' },
  { id: 41, score: 11200, linesCleared: 112, timestamp: '5 min ago' },
  { id: 40, score: 10500, linesCleared: 105, timestamp: '8 min ago' },
  { id: 39, score: 9800, linesCleared: 98, timestamp: '12 min ago' },
  { id: 38, score: 8900, linesCleared: 89, timestamp: '15 min ago' },
]

export default function TetrisAIPage() {
  const [isLearning, setIsLearning] = useState(false)
  const [isVisualizing, setIsVisualizing] = useState(false)
  const [selectedGeneration, setSelectedGeneration] = useState('42')

  const currentLearningGeneration = 43
  const latestCompletedGeneration = 42
  const bestScore = Math.max(...MOCK_GENERATIONS.map(g => g.score))

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background gradient orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[40%] -left-[20%] w-[80%] h-[80%] rounded-full bg-primary/10 blur-[120px] animate-pulse" />
        <div className="absolute -bottom-[40%] -right-[20%] w-[70%] h-[70%] rounded-full bg-accent/10 blur-[120px] animate-pulse" style={{ animationDelay: '1s' }} />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-6 max-w-7xl">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <div className="glass-card p-3 rounded-xl">
                <Brain className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground tracking-tight flex items-center gap-2">
                  Tetris AI
                  <Sparkles className="w-5 h-5 text-yellow-400" />
                </h1>
                <p className="text-sm text-muted-foreground">
                  Neural Network Training Visualization
                </p>
              </div>
            </div>

            <div className="glass-card px-4 py-2 rounded-xl">
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Best Score</p>
                  <p className="text-xl font-bold text-foreground font-mono">
                    {bestScore.toLocaleString()}
                  </p>
                </div>
                <div className="w-px h-8 bg-border" />
                <div className="text-right">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Generations</p>
                  <p className="text-xl font-bold text-foreground font-mono">
                    {latestCompletedGeneration}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column - Controls & Training Status */}
          <div className="lg:col-span-3 space-y-6">
            <ControlPanel
              isLearning={isLearning}
              isVisualizing={isVisualizing}
              currentLearningGeneration={currentLearningGeneration}
              latestCompletedGeneration={latestCompletedGeneration}
              selectedGeneration={selectedGeneration}
              onSelectedGenerationChange={setSelectedGeneration}
              onStartLearning={() => setIsLearning(true)}
              onStopLearning={() => setIsLearning(false)}
              onStartVisualize={() => setIsVisualizing(true)}
              onStopVisualize={() => setIsVisualizing(false)}
            />

            <TrainingStatus
              isLearning={isLearning}
              currentGeneration={currentLearningGeneration}
              latestCompletedGeneration={latestCompletedGeneration}
            />
          </div>

          {/* Center - Single Large Tetris Board */}
          <div className="lg:col-span-5">
            <div className="flex flex-col items-center gap-4">
              <TetrisBoard
                board={MOCK_BOARD}
                label={`Generation #${isVisualizing ? selectedGeneration : latestCompletedGeneration}`}
                activePiece={MOCK_ACTIVE_PIECE}
                ghostPosition={{ row: 13, col: 4 }}
                isActive={isVisualizing}
              />
              <div className="flex items-center gap-6">
                <NextPiece piece="L" />
                <StatsPanel
                  currentScore={12840}
                  highScore={bestScore}
                  linesCleared={128}
                  generation={latestCompletedGeneration}
                  isLearning={isLearning}
                  isVisualizing={isVisualizing}
                />
              </div>
            </div>
          </div>

          {/* Right Column - Analytics */}
          <div className="lg:col-span-4 space-y-6">
            <ScoreChart
              data={MOCK_CHART_DATA}
              title="Score Progress"
              description="Best & average scores per generation"
            />
            <GenerationList generations={MOCK_GENERATIONS} />
          </div>
        </div>
      </div>
    </div>
  )
}
