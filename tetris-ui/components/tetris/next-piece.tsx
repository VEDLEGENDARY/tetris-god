'use client'

import { cn } from '@/lib/utils'

// Tetris piece shapes (4x4 grid representation)
const PIECES: Record<string, number[][]> = {
  'I': [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ],
  'O': [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  'T': [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  'S': [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 0],
  ],
  'Z': [
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  'J': [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  'L': [
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0],
  ],
}

const PIECE_COLORS: Record<string, string> = {
  'I': 'bg-cyan-400',
  'O': 'bg-yellow-400',
  'T': 'bg-purple-400',
  'S': 'bg-green-400',
  'Z': 'bg-red-400',
  'J': 'bg-blue-400',
  'L': 'bg-orange-400',
}

const PIECE_GLOW: Record<string, string> = {
  'I': 'shadow-[0_0_8px_rgba(34,211,238,0.5)]',
  'O': 'shadow-[0_0_8px_rgba(250,204,21,0.5)]',
  'T': 'shadow-[0_0_8px_rgba(192,132,252,0.5)]',
  'S': 'shadow-[0_0_8px_rgba(74,222,128,0.5)]',
  'Z': 'shadow-[0_0_8px_rgba(248,113,113,0.5)]',
  'J': 'shadow-[0_0_8px_rgba(96,165,250,0.5)]',
  'L': 'shadow-[0_0_8px_rgba(251,146,60,0.5)]',
}

interface NextPieceProps {
  piece: string | null
}

export function NextPiece({ piece }: NextPieceProps) {
  const shape = piece ? PIECES[piece] : null

  return (
    <div className="glass-card rounded-2xl p-4">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Next Piece
        </h3>
      </div>
      
      <div className="flex items-center justify-center">
        <div className="grid grid-cols-4 gap-[2px] p-2">
          {shape ? (
            shape.map((row, rowIndex) =>
              row.map((cell, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={cn(
                    'w-4 h-4 rounded-sm transition-all',
                    cell === 1
                      ? cn(
                          PIECE_COLORS[piece!],
                          PIECE_GLOW[piece!],
                          'border border-white/30'
                        )
                      : 'bg-transparent'
                  )}
                />
              ))
            )
          ) : (
            Array(16).fill(0).map((_, i) => (
              <div
                key={i}
                className="w-4 h-4 rounded-sm bg-muted/20"
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}
