import React from 'react';
import { MapPin, Circle } from 'lucide-react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';

export function IntersectionMap() {
  const intersections = [
    { id: 'INT-001', name: 'Uhuru Highway & Kenyatta Ave', status: 'active', phase: 'NS-Green', queue: 3, delay: 18.2 },
    { id: 'INT-002', name: 'Moi Ave & Haile Selassie', status: 'active', phase: 'EW-Green', queue: 2, delay: 15.5 },
    { id: 'INT-003', name: 'Ngong Rd & Argwings Kodhek', status: 'active', phase: 'NS-Yellow', queue: 4, delay: 24.8 },
    { id: 'INT-004', name: 'Thika Rd & Outer Ring', status: 'active', phase: 'EW-Red', queue: 5, delay: 28.2 },
  ];

  const getPhaseColor = (phase: string) => {
    if (phase.includes('Green')) return 'bg-[#2ECC71]';
    if (phase.includes('Yellow')) return 'bg-[#F59E0B]';
    if (phase.includes('Red')) return 'bg-[#DC2626]';
    return 'bg-[#6B7280]';
  };

  const [selectedId, setSelectedId] = React.useState('INT-001');

  return (
    <Card className="p-6 border-[#E5E7EB] shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-[#1F2937]">Intersections</h3>
        <Badge variant="outline" className="border-[#2ECC71] text-[#2ECC71] bg-[#2ECC71]/5">
          4 Active
        </Badge>
      </div>

      {/* Map Placeholder */}
      <div className="mb-6 h-48 bg-[#F9FAFB] rounded-lg border-2 border-dashed border-[#E5E7EB] flex items-center justify-center">
        <div className="text-center text-[#9CA3AF]">
          <MapPin className="size-8 mx-auto mb-2" />
          <p>Map View (Nairobi CBD)</p>
        </div>
      </div>

      <div className="space-y-3">
        {intersections.map((intersection) => (
          <button
            key={intersection.id}
            onClick={() => setSelectedId(intersection.id)}
            className={`w-full text-left p-4 rounded-lg border transition-all duration-200 ${
              selectedId === intersection.id
                ? 'border-[#2ECC71] bg-[#2ECC71]/5 shadow-sm'
                : 'border-[#E5E7EB] hover:border-[#2ECC71]/50'
            }`}
            aria-pressed={selectedId === intersection.id}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-start gap-3 flex-1">
                <MapPin className="size-5 text-[#6B7280] mt-0.5 flex-shrink-0" />
                <div className="min-w-0">
                  <p className="text-[#1F2937] mb-1">{intersection.name}</p>
                  <p className="text-[#9CA3AF]">{intersection.id}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <Circle className={`size-2 ${getPhaseColor(intersection.phase)} rounded-full fill-current`} />
                <span className="text-[#6B7280]">{intersection.phase}</span>
              </div>
            </div>
            
            <div className="flex items-center gap-6 mt-3 pt-3 border-t border-[#E5E7EB]">
              <div>
                <p className="text-[#9CA3AF] mb-1">Queue</p>
                <p className="text-[#1F2937]">{intersection.queue} veh</p>
              </div>
              <div>
                <p className="text-[#9CA3AF] mb-1">Delay</p>
                <p className="text-[#1F2937]">{intersection.delay} s</p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </Card>
  );
}
