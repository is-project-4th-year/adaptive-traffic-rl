import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TimeSeriesChartProps {
  data: Array<{ time: string; rl: number; baseline: number }>;
}

export function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis 
          dataKey="time" 
          tick={{ fill: '#6B7280', fontSize: 12 }}
          stroke="#E5E7EB"
        />
        <YAxis 
          label={{ value: 'Delay (s)', angle: -90, position: 'insideLeft', fill: '#6B7280' }}
          tick={{ fill: '#6B7280', fontSize: 12 }}
          stroke="#E5E7EB"
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'white', 
            border: '1px solid #E5E7EB',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}
        />
        <Legend 
          wrapperStyle={{ paddingTop: '20px' }}
          iconType="line"
        />
        <Line 
          type="monotone" 
          dataKey="rl" 
          name="RL Controller"
          stroke="#2ECC71" 
          strokeWidth={2}
          dot={{ fill: '#2ECC71', r: 3 }}
        />
        <Line 
          type="monotone" 
          dataKey="baseline" 
          name="Baseline"
          stroke="#4B5563" 
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={{ fill: '#4B5563', r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
