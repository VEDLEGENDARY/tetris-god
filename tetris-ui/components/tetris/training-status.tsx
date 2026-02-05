'use client'

import { Brain, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface TrainingStatusProps {
  isLearning: boolean
  currentGeneration: number
  latestCompletedGeneration: number
}

export function TrainingStatus({
  isLearning,
  currentGeneration,
  latestCompletedGeneration,
}: TrainingStatusProps) {
  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="flex items-center gap-3">
        <div className={cn(
          'p-2.5 rounded-xl',
          isLearning ? 'bg-green-400/10' : 'bg-muted/20'
        )}>
          <Brain className={cn(
            'w-6 h-6',
            isLearning ? 'text-green-400' : 'text-muted-foreground'
          )} />
        </div>

        <div className="flex-1 min-w-0">
          {isLearning ? (
            <>
              <div className="flex items-center gap-2 mb-0.5">
                <Loader2 className="w-3.5 h-3.5 text-green-400 animate-spin" />
                <span className="text-xs font-medium text-green-400 uppercase tracking-wider">
                  Training
                </span>
              </div>
              <p className="text-lg font-bold font-mono text-foreground leading-tight">
                Gen #{currentGeneration}
              </p>
              <p className="text-xs text-muted-foreground">
                Last completed: #{latestCompletedGeneration}
              </p>
            </>
          ) : (
            <>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-0.5">
                Training Idle
              </p>
              <p className="text-sm font-mono text-muted-foreground">
                {latestCompletedGeneration > 0
                  ? `Last: Gen #${latestCompletedGeneration}`
                  : 'No training started'}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
