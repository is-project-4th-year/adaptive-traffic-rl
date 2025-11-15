import { Clock, CheckCircle2, AlertCircle, Info } from 'lucide-react';
import { Card } from './ui/card';

export function ActivityFeed() {
  const activities = [
    { 
      id: 1, 
      type: 'success', 
      message: 'Paired run #12 completed', 
      detail: 'RL improved avg delay by 28%',
      time: '2 min ago' 
    },
    { 
      id: 2, 
      type: 'success', 
      message: 'Episode #142 completed', 
      detail: 'Throughput +8% vs baseline',
      time: '15 min ago' 
    },
    { 
      id: 3, 
      type: 'info', 
      message: 'Model policy updated', 
      detail: 'DQN-v2.1.3 deployed',
      time: '28 min ago' 
    },
    { 
      id: 4, 
      type: 'warning', 
      message: 'High queue detected', 
      detail: 'Thika Rd & Outer Ring: 8 vehicles',
      time: '42 min ago' 
    },
    { 
      id: 5, 
      type: 'info', 
      message: 'Scheduled report sent', 
      detail: 'Daily summary to team',
      time: '1 hour ago' 
    },
  ];

  const getIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="size-4 text-[#059669]" />;
      case 'warning':
        return <AlertCircle className="size-4 text-[#F59E0B]" />;
      case 'info':
        return <Info className="size-4 text-[#2563EB]" />;
      default:
        return <Clock className="size-4 text-[#6B7280]" />;
    }
  };

  return (
    <Card className="p-6 border-[#E5E7EB]">
      <h3 className="text-[#1F2937] mb-6">Activity Feed</h3>
      
      <div className="space-y-4">
        {activities.map((activity) => (
          <div key={activity.id} className="flex gap-3">
            <div className="mt-0.5">{getIcon(activity.type)}</div>
            <div className="flex-1 min-w-0">
              <p className="text-[#1F2937] mb-1">{activity.message}</p>
              <p className="text-[#9CA3AF] mb-1">{activity.detail}</p>
              <p className="text-[#9CA3AF]">{activity.time}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}