'use client'

import { cn } from '@/lib/utils'
import { useMemo } from 'react'

const PIECE_COLORS: Record<string, { bg: string; border: string; glow: string }> = {
  'I': { bg: 'bg-cyan-400', border: 'border-cyan-300', glow: 'shadow-[0_0_16px_rgba(34,211,238,0.5)]' },
  'O': { bg: 'bg-yellow-400', border: 'border-yellow-300', glow: 'shadow-[0_0_16px_rgba(250,204,21,0.5)]' },
  'T': { bg: 'bg-purple-400', border: 'border-purple-300', glow: 'shadow-[0_0_16px_rgba(192,132,252,0.5)]' },
  'S': { bg: 'bg-green-400', border: 'border-green-300', glow: 'shadow-[0_0_16px_rgba(74,222,128,0.5)]' },
  'Z': { bg: 'bg-red-400', border: 'border-red-300', glow: 'shadow-[0_0_16px_rgba(248,113,113,0.5)]' },
  'J': { bg: 'bg-blue-400', border: 'border-blue-300', glow: 'shadow-[0_0_16px_rgba(96,165,250,0.5)]' },
  'L': { bg: 'bg-orange-400', border: 'border-orange-300', glow: 'shadow-[0_0_16px_rgba(251,146,60,0.5)]' },
}

const GHOST_MARKER = 'G'
const ACTIVE_MARKER = 'A'

interface TetrisBoardProps {
  board: (string | null)[][]
  label?: string
  isActive?: boolean
  activePiece?: {
    type: string
    position: { row: number; col: number }
    shape: number[][]
  } | null
  ghostPosition?: { row: number; col: number } | null
}

function getNeighborInfo(
  board: (string | null)[][],
  row: number,
  col: number,
  pieceType: string
): { top: boolean; right: boolean; bottom: boolean; left: boolean } {
  const rows = board.length
  const cols = board[0]?.length || 0

  const getBaseType = (cell: string | null) => {
    if (!cell) return null
    if (cell.startsWith(GHOST_MARKER)) return cell.slice(1)
    if (cell.startsWith(ACTIVE_MARKER)) return cell.slice(1)
    return cell
  }

  const currentBaseType = getBaseType(pieceType)
  const isCurrentActive = pieceType.startsWith(ACTIVE_MARKER)

  const isSamePiece = (r: number, c: number) => {
    if (r < 0 || r >= rows || c < 0 || c >= cols) return false
    const cell = board[r][c]
    if (!cell) return false
    const neighborBaseType = getBaseType(cell)
    const isNeighborActive = cell.startsWith(ACTIVE_MARKER)
    if (isCurrentActive) return isNeighborActive && neighborBaseType === currentBaseType
    return !cell.startsWith(GHOST_MARKER) && !cell.startsWith(ACTIVE_MARKER) && neighborBaseType === currentBaseType
  }

  return {
    top: isSamePiece(row - 1, col),
    right: isSamePiece(row, col + 1),
    bottom: isSamePiece(row + 1, col),
    left: isSamePiece(row, col - 1),
  }
}

function getBorderRadius(neighbors: { top: boolean; right: boolean; bottom: boolean; left: boolean }): string {
  const r = '6px'
  const n = '2px'
  return [
    !neighbors.top && !neighbors.left ? r : n,
    !neighbors.top && !neighbors.right ? r : n,
    !neighbors.bottom && !neighbors.right ? r : n,
    !neighbors.bottom && !neighbors.left ? r : n,
  ].join(' ')
}

export function TetrisBoard({
  board,
  label,
  isActive = false,
  activePiece = null,
  ghostPosition = null,
}: TetrisBoardProps) {
  const ROWS = 20
  const COLS = 10

  const displayBoard = useMemo(() => {
    const baseBoard = board.length > 0
      ? board.map(row => [...row])
      : Array(ROWS).fill(null).map(() => Array(COLS).fill(null))

    if (activePiece && ghostPosition) {
      const { type, shape } = activePiece
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (shape[r][c] === 1) {
            const br = ghostPosition.row + r
            const bc = ghostPosition.col + c
            if (br >= 0 && br < ROWS && bc >= 0 && bc < COLS && baseBoard[br][bc] === null) {
              baseBoard[br][bc] = `${GHOST_MARKER}${type}`
            }
          }
        }
      }
    }

    if (activePiece) {
      const { type, shape, position } = activePiece
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (shape[r][c] === 1) {
            const br = position.row + r
            const bc = position.col + c
            if (br >= 0 && br < ROWS && bc >= 0 && bc < COLS) {
              const cur = baseBoard[br][bc]
              if (cur === null || cur?.startsWith(GHOST_MARKER)) {
                baseBoard[br][bc] = `${ACTIVE_MARKER}${type}`
              }
            }
          }
        }
      }
    }

    return baseBoard
  }, [board, activePiece, ghostPosition])

  return (
    <div className="flex flex-col items-center gap-3">
      {label && (
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            {label}
          </span>
          {isActive && (
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
            </span>
          )}
        </div>
      )}
      <div className="glass-card rounded-2xl p-3 relative overflow-hidden">
        <div className="absolute inset-0 opacity-5">
          <div
            className="w-full h-full"
            style={{
              backgroundImage: 'linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)',
              backgroundSize: '30px 30px',
            }}
          />
        </div>

        <div
          className="grid gap-[2px] relative z-10"
          style={{
            gridTemplateColumns: `repeat(${COLS}, 1fr)`,
            gridTemplateRows: `repeat(${ROWS}, 1fr)`,
          }}
        >
          {displayBoard.map((row, rowIndex) =>
            row.map((cell, colIndex) => {
              const isGhostCell = cell?.startsWith(GHOST_MARKER)
              const isActiveCell = cell?.startsWith(ACTIVE_MARKER)
              const pieceType = isGhostCell ? cell?.slice(1) : isActiveCell ? cell?.slice(1) : cell
              const hasLandedBlock = cell !== null && !isGhostCell && !isActiveCell
              const colorInfo = pieceType ? PIECE_COLORS[pieceType] : null

              const neighbors = (hasLandedBlock || isActiveCell) && cell
                ? getNeighborInfo(displayBoard, rowIndex, colIndex, cell)
                : { top: false, right: false, bottom: false, left: false }

              const borderRadius = (hasLandedBlock || isActiveCell)
                ? getBorderRadius(neighbors)
                : '4px'

              if (isGhostCell && colorInfo) {
                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className="w-7 h-7 border-2 border-white/30 bg-white/5 transition-all duration-200 ease-out"
                    style={{ borderRadius: '50%' }}
                  />
                )
              }

              if ((isActiveCell || hasLandedBlock) && colorInfo) {
                return (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className={cn(
                      'w-7 h-7 relative overflow-hidden transition-all duration-200 ease-out',
                      colorInfo.bg,
                      colorInfo.glow,
                    )}
                    style={{ borderRadius }}
                  >
                    <div
                      className="absolute inset-0 bg-gradient-to-br from-white/40 via-transparent to-black/20"
                      style={{ borderRadius }}
                    />
                    {!neighbors.top && (
                      <div
                        className="absolute top-0 left-0 right-0 h-[2px] bg-white/50"
                        style={{
                          borderTopLeftRadius: !neighbors.left ? '4px' : '0',
                          borderTopRightRadius: !neighbors.right ? '4px' : '0',
                        }}
                      />
                    )}
                    {!neighbors.left && (
                      <div
                        className="absolute top-0 bottom-0 left-0 w-[2px] bg-white/30"
                        style={{
                          borderTopLeftRadius: !neighbors.top ? '4px' : '0',
                          borderBottomLeftRadius: !neighbors.bottom ? '4px' : '0',
                        }}
                      />
                    )}
                  </div>
                )
              }

              return (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="w-7 h-7 rounded bg-muted/10 border border-muted/5 transition-all duration-150"
                />
              )
            })
          )}
        </div>

        <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent pointer-events-none rounded-2xl" />
      </div>
    </div>
  )
}
