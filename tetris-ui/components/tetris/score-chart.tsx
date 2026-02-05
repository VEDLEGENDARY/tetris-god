'use client'

import {
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts'
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart'

interface ScoreChartProps {
  data: { generation: number; score: number; avgScore?: number }[]
  title: string
  description?: string
}

export function ScoreChart({ data, title, description }: ScoreChartProps) {
  // Compute colors in JavaScript for Recharts
  const primaryColor = '#6366f1' // indigo-500
  const accentColor = '#22d3ee' // cyan-400

  return (
    <div className="glass-card rounded-2xl p-5 h-full">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-foreground">{title}</h3>
        {description && (
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
        )}
      </div>
      
      <ChartContainer
        config={{
          score: {
            label: 'Best Score',
            color: primaryColor,
          },
          avgScore: {
            label: 'Avg Score',
            color: accentColor,
          },
        }}
        className="h-[200px] w-full"
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={primaryColor} stopOpacity={0.3} />
                <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="avgGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={accentColor} stopOpacity={0.2} />
                <stop offset="95%" stopColor={accentColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="rgba(255,255,255,0.05)" 
              vertical={false}
            />
            <XAxis 
              dataKey="generation" 
              stroke="rgba(255,255,255,0.3)"
              fontSize={11}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="rgba(255,255,255,0.3)"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value}
            />
            <ChartTooltip 
              content={<ChartTooltipContent />}
              cursor={{ stroke: 'rgba(255,255,255,0.1)' }}
            />
            <Area
              type="monotone"
              dataKey="avgScore"
              stroke={accentColor}
              strokeWidth={2}
              fill="url(#avgGradient)"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="score"
              stroke={primaryColor}
              strokeWidth={2}
              fill="url(#scoreGradient)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  )
}
