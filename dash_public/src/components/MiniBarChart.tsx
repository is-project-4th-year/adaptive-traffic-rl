import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Legend, Cell } from 'recharts';

interface MiniBarChartProps {
  data: Array<{ metric: string; rl: number; baseline: number }>;
  dataKeyRL: string;
  dataKeyBaseline: string;
}

export function MiniBarChart({ data, dataKeyRL, dataKeyBaseline }: MiniBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
        <XAxis 
          dataKey="metric" 
          tick={{ fill: '#6B7280', fontSize: 12 }}
          axisLine={{ stroke: '#E5E7EB' }}
        />
        <YAxis 
          tick={{ fill: '#6B7280', fontSize: 12 }}
          axisLine={{ stroke: '#E5E7EB' }}
          width={40}
        />
        <Legend 
          wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
          iconType="circle"
        />
        <Bar 
          dataKey={dataKeyRL} 
          name="RL" 
          fill="#2ECC71" 
          radius={[4, 4, 0, 0]}
          maxBarSize={40}
        />
        <Bar 
          dataKey={dataKeyBaseline} 
          name="Baseline" 
          fill="#4B5563" 
          radius={[4, 4, 0, 0]}
          maxBarSize={40}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
