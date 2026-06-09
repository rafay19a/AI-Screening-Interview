import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Building2,
  Bed,
  CalendarDays,
  Users,
  Wallet,
  Settings
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Properties', href: '/properties', icon: Building2 },
  { name: 'Units', href: '/units', icon: Bed },
  { name: 'Bookings', href: '/bookings', icon: CalendarDays },
  { name: 'Occupancy', href: '/occupancy', icon: CalendarDays },
  { name: 'Guests', href: '/guests', icon: Users },
  { name: 'Finance', href: '/finance', icon: Wallet },
  { name: 'Housekeeping', href: '/housekeeping', icon: Users },
  { name: 'Maintenance', href: '/maintenance', icon: Users },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export const Sidebar = () => {
  return (
    <div className="flex flex-col w-64 bg-white border-r border-gray-200">
      <div className="flex items-center h-16 px-6 border-b border-gray-200">
        <span className="text-xl font-bold text-primary">ResortERP</span>
      </div>
      <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                'flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              )
            }
          >
            <item.icon className="w-5 h-5 mr-3" />
            {item.name}
          </NavLink>
        ))}
      </nav>
    </div>
  );
};
