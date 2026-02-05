'use client'

import { Trophy, Target, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'

interface StatsPanelProps {
  currentScore: number
  highScore: number
  linesCleared: number
  generation: number
  isLearning: boolean
  isVisualizing: boolean
}

export function StatsPanel({
  currentScore,
  highScore,
  linesCleared,
  generation,
}: StatsPanelProps) {
  const stats = [
    {
      label: 'Score',
      value: currentScore.toLocaleString(),
      icon: Target,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-400/10',
    },
    {
      label: 'High Score',
      value: highScore.toLocaleString(),
      icon: Trophy,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-400/10',
    },
    {
      label: 'Lines',
      value: linesCleared.toLocaleString(),
      icon: Zap,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10',
    },
  ]

  return (
    <div className="glass-card rounded-2xl p-4">
      <p className="text-xs text-muted-foreground uppercase tracking-wider mb-3">
        Gen #{generation}
      </p>
      <div className="flex flex-col gap-3">
        {stats.map((stat) => (
          <div key={stat.label} className="flex items-center gap-2">
            <div className={cn('p-1.5 rounded-lg', stat.bgColor)}>
              <stat.icon className={cn('w-3.5 h-3.5', stat.color)} />
            </div>
            <div className="flex-1 min-w-0">
              <span className="text-xs text-muted-foreground">{stat.label}</span>
              <p className="text-sm font-bold font-mono text-foreground leading-tight">
                {stat.value}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
