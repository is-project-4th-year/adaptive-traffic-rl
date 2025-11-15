import { Play, Trash2, Key, Lock, Upload } from 'lucide-react';
import { useState } from 'react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';

interface AdminReportsProps {
  userRole: 'admin' | 'engineer' | 'viewer';
}

export function AdminReports({ userRole }: AdminReportsProps) {
  const isAdmin = userRole === 'admin';
  const [showClearDialog, setShowClearDialog] = useState(false);

  const users = [
    { name: 'Alice Johnson', role: 'Admin', email: 'alice@example.com', lastActive: '2 hours ago' },
    { name: 'Bob Smith', role: 'Engineer', email: 'bob@example.com', lastActive: '30 min ago' },
    { name: 'Carol White', role: 'Viewer', email: 'carol@example.com', lastActive: '1 day ago' },
  ];

  const policyVersions = [
    { version: 'DQN-v2.1.0', deployed: 'Nov 11, 2025', episodes: 45, status: 'Active' },
    { version: 'DQN-v2.0.8', deployed: 'Nov 8, 2025', episodes: 128, status: 'Archived' },
    { version: 'DQN-v2.0.5', deployed: 'Nov 1, 2025', episodes: 87, status: 'Archived' },
  ];

  const auditLog = [
    { timestamp: 'Nov 11, 10:45 AM', user: 'Alice Johnson', action: 'Started paired run #143', status: 'success' },
    { timestamp: 'Nov 11, 09:30 AM', user: 'Bob Smith', action: 'Exported episode data', status: 'success' },
    { timestamp: 'Nov 11, 08:15 AM', user: 'Alice Johnson', action: 'Uploaded policy DQN-v2.1.0', status: 'success' },
    { timestamp: 'Nov 10, 04:22 PM', user: 'Bob Smith', action: 'Modified intersection config', status: 'warning' },
  ];

  if (!isAdmin) {
    return (
      <div className="p-8 max-w-[1440px] mx-auto">
        <div className="flex items-center justify-center h-96">
          <Card className="p-8 text-center border-[#E5E7EB]">
            <Lock className="size-12 text-[#6B7280] mx-auto mb-4" />
            <h2 className="text-[#1F2937] mb-2">Admin Access Required</h2>
            <p className="text-[#6B7280]">
              You need administrator privileges to access this section.
            </p>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-[1440px] mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-[#1F2937] mb-1">Admin & Reports</h1>
          <p className="text-[#6B7280]">System management and governance</p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <Card className="p-6 border-[#E5E7EB] hover:border-[#2ECC71] hover:shadow-md transition-all duration-200 cursor-pointer">
          <Play className="size-8 text-[#2ECC71] mb-3" />
          <h3 className="text-[#1F2937] mb-1">Run Evaluation</h3>
          <p className="text-[#6B7280]">Start new comparison</p>
        </Card>

        <Card className="p-6 border-[#E5E7EB] hover:border-[#2563EB] hover:shadow-md transition-all duration-200 cursor-pointer">
          <div className="flex items-center gap-2 mb-3">
            <Upload className="size-8 text-[#2563EB]" />
            <Lock className="size-4 text-[#6B7280]" aria-label="Admin only" />
          </div>
          <h3 className="text-[#1F2937] mb-1">Upload Policy</h3>
          <p className="text-[#6B7280]">Deploy new model</p>
        </Card>

        <button
          onClick={() => setShowClearDialog(true)}
          className="text-left"
        >
          <Card className="p-6 border-[#E5E7EB] hover:border-[#F59E0B] hover:shadow-md transition-all duration-200 cursor-pointer">
            <Trash2 className="size-8 text-[#F59E0B] mb-3" />
            <h3 className="text-[#1F2937] mb-1">Clear Temp</h3>
            <p className="text-[#6B7280]">Free up storage</p>
          </Card>
        </button>

        <Card className="p-6 border-[#E5E7EB] hover:border-[#DC2626] hover:shadow-md transition-all duration-200 cursor-pointer">
          <Key className="size-8 text-[#DC2626] mb-3" />
          <h3 className="text-[#1F2937] mb-1">Rotate Keys</h3>
          <p className="text-[#6B7280]">Security update</p>
        </Card>
      </div>

      {/* Users & Roles */}
      <Card className="p-6 border-[#E5E7EB] shadow-sm mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-[#1F2937]">Users & Roles</h3>
          <Button variant="outline" size="sm">Add User</Button>
        </div>
        <div className="rounded-lg border border-[#E5E7EB] overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Email</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Last Active</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {users.map((user) => (
                <TableRow key={user.email}>
                  <TableCell>{user.name}</TableCell>
                  <TableCell className="text-[#6B7280]">{user.email}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{user.role}</Badge>
                  </TableCell>
                  <TableCell className="text-[#6B7280]">{user.lastActive}</TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="sm">Edit</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Policy Versions & Data Retention */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <Card className="p-6 border-[#E5E7EB] shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[#1F2937]">Policy Versions</h3>
            <Button variant="outline" size="sm" className="gap-2">
              <Upload className="size-4" />
              <Lock className="size-3" aria-label="Admin only" />
              Upload
            </Button>
          </div>
          <div className="rounded-lg border border-[#E5E7EB] overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Episodes</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {policyVersions.map((policy) => (
                  <TableRow key={policy.version}>
                    <TableCell>{policy.version}</TableCell>
                    <TableCell>{policy.episodes}</TableCell>
                    <TableCell>
                      <Badge 
                        variant="outline"
                        className={
                          policy.status === 'Active'
                            ? 'border-[#2ECC71] text-[#2ECC71] bg-[#2ECC71]/5'
                            : 'border-[#6B7280] text-[#6B7280]'
                        }
                      >
                        {policy.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Card>

        <Card className="p-6 border-[#E5E7EB] shadow-sm">
          <h3 className="text-[#1F2937] mb-4">Data Retention</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-[#F9FAFB] rounded-lg">
              <div>
                <p className="text-[#1F2937] mb-1">Episode Data</p>
                <p className="text-[#6B7280]">Keep for 90 days</p>
              </div>
              <Button variant="outline" size="sm">Edit</Button>
            </div>
            <div className="flex items-center justify-between p-3 bg-[#F9FAFB] rounded-lg">
              <div>
                <p className="text-[#1F2937] mb-1">Logs & Audit</p>
                <p className="text-[#6B7280]">Keep for 1 year</p>
              </div>
              <Button variant="outline" size="sm">Edit</Button>
            </div>
            <div className="flex items-center justify-between p-3 bg-[#F9FAFB] rounded-lg">
              <div>
                <p className="text-[#1F2937] mb-1">Temp Files</p>
                <p className="text-[#6B7280]">47 files (2.3 GB)</p>
              </div>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowClearDialog(true)}
              >
                Clear
              </Button>
            </div>
          </div>
        </Card>
      </div>

      {/* Audit Log */}
      <Card className="p-6 border-[#E5E7EB] shadow-sm">
        <h3 className="text-[#1F2937] mb-4">Audit Log</h3>
        <div className="rounded-lg border border-[#E5E7EB] overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp</TableHead>
                <TableHead>User</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {auditLog.map((log, index) => (
                <TableRow key={index}>
                  <TableCell className="text-[#6B7280]">{log.timestamp}</TableCell>
                  <TableCell>{log.user}</TableCell>
                  <TableCell>{log.action}</TableCell>
                  <TableCell>
                    <Badge 
                      variant="outline"
                      className={
                        log.status === 'success'
                          ? 'border-[#059669] text-[#059669] bg-[#059669]/5'
                          : 'border-[#F59E0B] text-[#F59E0B] bg-[#F59E0B]/5'
                      }
                    >
                      {log.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Clear Confirmation Dialog */}
      <Dialog open={showClearDialog} onOpenChange={setShowClearDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Clear 47 temp files?</DialogTitle>
            <DialogDescription className="pt-2">
              This will permanently delete 2.3 GB of temporary files. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              variant="outline"
              onClick={() => setShowClearDialog(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-[#DC2626] hover:bg-[#B91C1C] text-white"
              onClick={() => {
                // Handle clear action
                setShowClearDialog(false);
              }}
            >
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}