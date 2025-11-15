export function BoxPlotChart() {
  const boxPlotData = [
    { 
      label: 'RL Delay',
      min: 12,
      q1: 16,
      median: 20.5,
      q3: 24,
      max: 28,
      color: '#2ECC71'
    },
    { 
      label: 'Baseline Delay',
      min: 22,
      q1: 28,
      median: 32.8,
      q3: 38,
      max: 45,
      color: '#4B5563'
    },
  ];

  const maxValue = 50;

  return (
    <div className="space-y-12 py-6">
      {boxPlotData.map((data) => (
        <div key={data.label}>
          <p className="text-[#6B7280] mb-4">{data.label}</p>
          <div className="relative h-16">
            {/* Scale */}
            <div className="absolute bottom-0 left-0 right-0 flex justify-between text-[#9CA3AF] mb-2">
              <span>0</span>
              <span>25s</span>
              <span>50s</span>
            </div>
            
            {/* Box plot */}
            <div className="relative h-12">
              {/* Whisker line */}
              <div 
                className="absolute top-6 h-px bg-current"
                style={{ 
                  left: `${(data.min / maxValue) * 100}%`,
                  width: `${((data.max - data.min) / maxValue) * 100}%`,
                  color: data.color
                }}
              />
              
              {/* Box */}
              <div 
                className="absolute top-3 h-6 rounded"
                style={{ 
                  left: `${(data.q1 / maxValue) * 100}%`,
                  width: `${((data.q3 - data.q1) / maxValue) * 100}%`,
                  backgroundColor: `${data.color}20`,
                  border: `2px solid ${data.color}`
                }}
              />
              
              {/* Median line */}
              <div 
                className="absolute top-2 h-8 w-0.5"
                style={{ 
                  left: `${(data.median / maxValue) * 100}%`,
                  backgroundColor: data.color
                }}
              />
              
              {/* Min/Max markers */}
              <div 
                className="absolute top-4 w-px h-4"
                style={{ 
                  left: `${(data.min / maxValue) * 100}%`,
                  backgroundColor: data.color
                }}
              />
              <div 
                className="absolute top-4 w-px h-4"
                style={{ 
                  left: `${(data.max / maxValue) * 100}%`,
                  backgroundColor: data.color
                }}
              />
            </div>
          </div>
          
          <div className="flex gap-6 mt-4 text-[#6B7280]">
            <span>Min: {data.min}s</span>
            <span>Q1: {data.q1}s</span>
            <span>Median: {data.median}s</span>
            <span>Q3: {data.q3}s</span>
            <span>Max: {data.max}s</span>
          </div>
        </div>
      ))}
    </div>
  );
}
