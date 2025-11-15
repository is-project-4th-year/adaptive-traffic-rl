import {
  TrafficCone,
  LayoutDashboard,
  Activity,
  List,
  Settings,
  User,
} from "lucide-react";

import { NavLink } from "react-router-dom";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";

interface AppShellProps {
  userRole: "admin" | "engineer" | "viewer";
}

export function AppShell({ userRole }: AppShellProps) {
  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, href: "/" },
    { id: "live", label: "Live Compare", icon: Activity, href: "/live" },
    { id: "episodes", label: "Episodes", icon: List, href: "/episodes" },
    {
      id: "admin",
      label: "Admin",
      icon: Settings,
      href: "/admin",
      adminOnly: true,
    },
  ];

  const roleColors = {
    admin: "bg-[#DC2626] text-white",
    engineer: "bg-[#3B82F6] text-white",
    viewer: "bg-[#64748B] text-white",
  };

  return (
    <>
      {/* ------------------------------------------------------
         TOP BAR 
      ------------------------------------------------------ */}
      <header className="fixed top-0 left-0 right-0 h-16 bg-white border-b border-[#E2E8F0] z-50 shadow-sm">
        <div className="h-full px-6 flex items-center justify-between">
          {/* BRAND */}
          <div className="flex items-center gap-3">
            <TrafficCone className="size-8 text-[#059669]" />
            <span className="text-[#0F172A] text-lg font-medium">
              TrafficRL
            </span>

            <Badge
              variant="outline"
              className="ml-2 border-[#3B82F6] text-[#3B82F6] bg-[#EFF6FF]"
            >
              Production
            </Badge>
          </div>

          {/* USER MENU */}
          <div className="flex items-center gap-4">
            <Badge className={roleColors[userRole]}>
              {userRole.charAt(0).toUpperCase() + userRole.slice(1)}
            </Badge>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex items-center gap-2 focus:ring-2 focus:ring-[#059669]"
                >
                  <div className="size-8 rounded-full bg-gradient-to-br from-[#059669] to-[#047857] flex items-center justify-center">
                    <User className="size-4 text-white" />
                  </div>
                  <span className="text-[#0F172A]">User Name</span>
                </Button>
              </DropdownMenuTrigger>

              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>Profile</DropdownMenuItem>
                <DropdownMenuItem>Settings</DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem>Logout</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------
         SIDEBAR NAVIGATION 
      ------------------------------------------------------ */}
      <aside className="fixed left-0 top-16 bottom-0 w-64 bg-white border-r border-[#E2E8F0] z-40 shadow-sm">
        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            if (item.adminOnly && userRole !== "admin") return null;

            const Icon = item.icon;

            return (
              <NavLink
                key={item.id}
                to={item.href}
                aria-label={item.label}
                className={({ isActive }) =>
                  `w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 outline-none ${
                    isActive
                      ? "bg-gradient-to-r from-[#059669] to-[#047857] text-white shadow-md"
                      : "text-[#475569] hover:bg-[#F8FAFC] hover:text-[#0F172A]"
                  }`
                }
              >
                <Icon className="size-5" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </aside>
    </>
  );
}

export default AppShell;
