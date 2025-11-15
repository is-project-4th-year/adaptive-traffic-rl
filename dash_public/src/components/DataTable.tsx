import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table';

interface DataTableProps {
  data: Array<{
    approach: string;
    rlDelay: number;
    baselineDelay: number;
    rlQueue: number;
    baselineQueue: number;
  }>;
}

export function DataTable({ data }: DataTableProps) {
  return (
    <div className="rounded-lg border border-[#E5E7EB] overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Approach</TableHead>
            <TableHead className="text-right">RL Delay (s)</TableHead>
            <TableHead className="text-right">Base Delay (s)</TableHead>
            <TableHead className="text-right">RL Queue</TableHead>
            <TableHead className="text-right">Base Queue</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row) => (
            <TableRow key={row.approach}>
              <TableCell>{row.approach}</TableCell>
              <TableCell className="text-right text-[#2ECC71]">{row.rlDelay}</TableCell>
              <TableCell className="text-right text-[#4B5563]">{row.baselineDelay}</TableCell>
              <TableCell className="text-right text-[#2ECC71]">{row.rlQueue}</TableCell>
              <TableCell className="text-right text-[#4B5563]">{row.baselineQueue}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
