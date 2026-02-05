'use client'

import { Play, Eye, Square, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

interface ControlPanelProps {
  isLearning: boolean
  isVisualizing: boolean
  currentLearningGeneration: number
  latestCompletedGeneration: number
  selectedGeneration: string
  onSelectedGenerationChange: (value: string) => void
  onStartLearning: () => void
  onStopLearning: () => void
  onStartVisualize: () => void
  onStopVisualize: () => void
}

export function ControlPanel({
  isLearning,
  isVisualizing,
  currentLearningGeneration,
  latestCompletedGeneration,
  selectedGeneration,
  onSelectedGenerationChange,
  onStartLearning,
  onStopLearning,
  onStartVisualize,
  onStopVisualize,
}: ControlPanelProps) {
  const isValidGeneration = () => {
    const gen = parseInt(selectedGeneration)
    return !isNaN(gen) && gen >= 1 && gen <= latestCompletedGeneration
  }

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-foreground">Controls</h3>
        <p className="text-sm text-muted-foreground mt-1">Manage AI training and visualization</p>
      </div>
      
      <div className="space-y-4">
        {/* Learning Control */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Training
            </label>
            {isLearning && (
              <span className="text-xs text-green-400 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Learning Gen #{currentLearningGeneration}
              </span>
            )}
          </div>
          <Button
            onClick={isLearning ? onStopLearning : onStartLearning}
            className={cn(
              'w-full h-12 text-base font-semibold transition-all duration-200',
              isLearning
                ? 'bg-red-500/90 hover:bg-red-500 text-white'
                : 'glass-button text-white'
            )}
          >
            {isLearning ? (
              <>
                <Square className="w-5 h-5 mr-2 fill-current" />
                Stop Learning
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2 fill-current" />
                Start Learning
              </>
            )}
          </Button>
        </div>
        
        {/* Generation Selection */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Visualize Gen
            </label>
            <span className="text-xs text-muted-foreground">
              Latest: <span className="text-primary font-mono font-bold">{latestCompletedGeneration}</span>
            </span>
          </div>
          <div className="relative">
            <Input
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={4}
              value={selectedGeneration}
              onChange={(e) => {
                const value = e.target.value.replace(/\D/g, '').slice(0, 4)
                onSelectedGenerationChange(value)
              }}
              placeholder="0001"
              className={cn(
                'glass-input h-12 text-center font-mono text-xl tracking-widest',
                !isValidGeneration() && selectedGeneration && 'border-red-500/50 focus:border-red-500'
              )}
            />
            {!isValidGeneration() && selectedGeneration && (
              <p className="text-xs text-red-400 mt-1">
                Enter a number between 1 and {latestCompletedGeneration}
              </p>
            )}
          </div>
        </div>
        
        {/* Visualization Control */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Visualization
            </label>
            {isVisualizing && (
              <span className="text-xs text-blue-400 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                Showing Gen #{selectedGeneration}
              </span>
            )}
          </div>
          <Button
            onClick={isVisualizing ? onStopVisualize : onStartVisualize}
            disabled={!isVisualizing && !isValidGeneration()}
            variant="outline"
            className={cn(
              'w-full h-12 text-base font-semibold transition-all duration-200',
              isVisualizing
                ? 'bg-red-500/90 hover:bg-red-500 text-white border-red-500'
                : 'border-primary/50 hover:bg-primary/10 text-primary'
            )}
          >
            {isVisualizing ? (
              <>
                <Square className="w-5 h-5 mr-2 fill-current" />
                Stop Visualization
              </>
            ) : (
              <>
                <Eye className="w-5 h-5 mr-2" />
                Start Visual
              </>
            )}
          </Button>
        </div>
        
        {/* Status Indicator */}
        <div className="pt-3 border-t border-border/50 space-y-2">
          <div className="flex items-center gap-2">
            <div className={cn(
              'w-2 h-2 rounded-full',
              isLearning ? 'bg-green-500 animate-pulse' : 'bg-muted-foreground/50'
            )} />
            <span className="text-sm text-muted-foreground">
              {isLearning ? (
                <span className="flex items-center gap-2 text-green-400">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Training Gen #{currentLearningGeneration}
                </span>
              ) : (
                'Training idle'
              )}
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <div className={cn(
              'w-2 h-2 rounded-full',
              isVisualizing ? 'bg-blue-500 animate-pulse' : 'bg-muted-foreground/50'
            )} />
            <span className="text-sm text-muted-foreground">
              {isVisualizing ? (
                <span className="text-blue-400">Visualizing Gen #{selectedGeneration}</span>
              ) : (
                'Visualization idle'
              )}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
