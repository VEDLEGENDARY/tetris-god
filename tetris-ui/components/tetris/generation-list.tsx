'use client'

import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Generation {
  id: number
  score: number
  linesCleared: number
  timestamp: string
}

interface GenerationListProps {
  generations: Generation[]
}

export function GenerationList({ generations }: GenerationListProps) {
  return (
    <div className="glass-card rounded-2xl p-5 h-full">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-foreground">Recent Generations</h3>
        <p className="text-sm text-muted-foreground mt-1">Last 5 completed runs</p>
      </div>
      
      <div className="space-y-2">
        {generations.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground text-sm">
            No generations completed yet
          </div>
        ) : (
          generations.map((gen, index) => {
            const prevGen = generations[index + 1]
            const scoreDiff = prevGen ? gen.score - prevGen.score : 0
            const isImprovement = scoreDiff > 0
            const isDecline = scoreDiff < 0
            
            return (
              <div
                key={gen.id}
                className={cn(
                  'flex items-center justify-between p-3 rounded-xl transition-all',
                  'bg-muted/30 hover:bg-muted/50',
                  index === 0 && 'ring-1 ring-primary/30 bg-primary/5'
                )}
              >
                <div className="flex items-center gap-3">
                  <div className={cn(
                    'w-10 h-10 rounded-lg flex items-center justify-center font-mono text-sm font-bold',
                    index === 0 
                      ? 'bg-primary/20 text-primary' 
                      : 'bg-muted/50 text-muted-foreground'
                  )}>
                    #{gen.id}
                  </div>
                  <div>
                    <div className="font-semibold text-foreground">
                      {gen.score.toLocaleString()} pts
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {gen.linesCleared} lines • {gen.timestamp}
                    </div>
                  </div>
                </div>
                
                <div className={cn(
                  'flex items-center gap-1 text-sm font-medium',
                  isImprovement && 'text-green-400',
                  isDecline && 'text-red-400',
                  !isImprovement && !isDecline && 'text-muted-foreground'
                )}>
                  {isImprovement && <TrendingUp className="w-4 h-4" />}
                  {isDecline && <TrendingDown className="w-4 h-4" />}
                  {!isImprovement && !isDecline && prevGen && <Minus className="w-4 h-4" />}
                  {prevGen && (
                    <span>
                      {isImprovement && '+'}
                      {scoreDiff.toLocaleString()}
                    </span>
                  )}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
