import { useQuery } from '@tanstack/react-query';
import { supabase } from '../../lib/supabase';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, isSameDay } from 'date-fns';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useState } from 'react';

export const OccupancyPage = () => {
  const [currentMonth, setCurrentMonth] = useState(new Date());

  const monthStart = startOfMonth(currentMonth);
  const monthEnd = endOfMonth(currentMonth);
  const days = eachDayOfInterval({ start: monthStart, end: monthEnd });

  const { data: units } = useQuery({
    queryKey: ['units-occupancy'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('units')
        .select('id, name, properties(name)');
      if (error) throw error;
      return data;
    },
  });

  const { data: bookings } = useQuery({
    queryKey: ['bookings-occupancy', currentMonth],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('bookings')
        .select('*, booking_units(unit_id)')
        .gte('check_out', monthStart.toISOString())
        .lte('check_in', monthEnd.toISOString());
      if (error) throw error;
      return data;
    },
  });

  const isUnitOccupied = (unitId: string, day: Date) => {
    return bookings?.find(booking =>
      booking.booking_units?.some((bu: any) => bu.unit_id === unitId) &&
      new Date(booking.check_in) <= day &&
      new Date(booking.check_out) > day
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Occupancy Matrix</h1>
          <p className="text-sm text-gray-500">Visual availability and stay tracking</p>
        </div>
        <div className="flex items-center bg-white border border-gray-300 rounded-md p-1">
          <button
            onClick={() => setCurrentMonth(new Date(currentMonth.setMonth(currentMonth.getMonth() - 1)))}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <span className="px-4 text-sm font-medium">{format(currentMonth, 'MMMM yyyy')}</span>
          <button
            onClick={() => setCurrentMonth(new Date(currentMonth.setMonth(currentMonth.getMonth() + 1)))}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="bg-white border border-gray-100 rounded-xl shadow-sm overflow-x-auto">
        <div className="min-w-max">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 bg-gray-50 p-4 border-b border-r border-gray-200 text-left text-xs font-semibold text-gray-500 uppercase min-w-[200px]">
                  Unit / Room
                </th>
                {days.map(day => (
                  <th key={day.toString()} className={
                    `p-2 border-b border-gray-200 text-center text-[10px] font-medium min-w-[32px]
                    ${isSameDay(day, new Date()) ? 'bg-primary/5 text-primary' : 'text-gray-500'}`
                  }>
                    {format(day, 'd')}
                    <div className="text-[8px] font-normal">{format(day, 'EEE')}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {units?.map((unit: any) => (
                <tr key={unit.id} className="hover:bg-gray-50">
                  <td className="sticky left-0 z-10 bg-white p-4 border-r border-b border-gray-100">
                    <div className="text-sm font-medium text-gray-900">{unit.name}</div>
                    <div className="text-[10px] text-gray-400">{unit.properties?.name}</div>
                  </td>
                  {days.map(day => {
                    const booking = isUnitOccupied(unit.id, day);
                    return (
                      <td
                        key={day.toString()}
                        className={`border-b border-gray-100 p-1 relative min-h-[40px] ${isSameDay(day, new Date()) ? 'bg-primary/5' : ''}`}
                      >
                        {booking && (
                          <div className={`
                            absolute inset-y-1 inset-x-0 rounded-sm text-[8px] p-1 flex items-center overflow-hidden
                            ${booking.status === 'confirmed' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}
                          `}>
                            {isSameDay(day, new Date(booking.check_in)) && (
                              <span className="truncate whitespace-nowrap">Guest</span>
                            )}
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex items-center space-x-6 text-xs text-gray-500">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-100 border border-green-200 rounded-sm mr-2"></div>
          Confirmed
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-100 border border-blue-200 rounded-sm mr-2"></div>
          Checked In
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-gray-100 border border-gray-200 rounded-sm mr-2"></div>
          Maintenance
        </div>
      </div>
    </div>
  );
};
