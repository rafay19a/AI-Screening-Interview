import { useQuery } from '@tanstack/react-query';
import { supabase } from '../../lib/supabase';
import { Users as Broom, Plus, CheckCircle, Clock, AlertTriangle } from 'lucide-react';

export const HousekeepingPage = () => {
  const { data: tasks, isLoading } = useQuery({
    queryKey: ['housekeeping'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('housekeeping_tasks')
        .select('*, units(name, properties(name))')
        .order('created_at', { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Housekeeping</h1>
          <p className="text-sm text-gray-500">Manage cleaning tasks and unit readiness</p>
        </div>
        <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary hover:bg-primary/90 shadow-sm">
          <Plus className="w-4 h-4 mr-2" />
          Assign Task
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm flex items-center">
          <div className="h-10 w-10 bg-yellow-50 rounded-lg flex items-center justify-center mr-4 text-yellow-600">
            <Clock className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold">12</div>
            <div className="text-xs text-gray-500 font-medium">Pending Tasks</div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm flex items-center">
          <div className="h-10 w-10 bg-blue-50 rounded-lg flex items-center justify-center mr-4 text-blue-600">
            <Broom className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold">5</div>
            <div className="text-xs text-gray-500 font-medium">In Progress</div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm flex items-center">
          <div className="h-10 w-10 bg-green-50 rounded-lg flex items-center justify-center mr-4 text-green-600">
            <CheckCircle className="w-6 h-6" />
          </div>
          <div>
            <div className="text-2xl font-bold">28</div>
            <div className="text-xs text-gray-500 font-medium">Completed Today</div>
          </div>
        </div>
      </div>

      <div className="bg-white shadow-sm border border-gray-100 rounded-xl overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Unit</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Property</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Priority</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {isLoading ? (
              <tr><td colSpan={5} className="py-12 text-center">Loading...</td></tr>
            ) : (
              tasks?.map((task: any) => (
                <tr key={task.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{task.units?.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{task.units?.properties?.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 text-xs font-medium rounded-full bg-yellow-100 text-yellow-800">
                      {task.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-600">{task.priority}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <button className="text-primary hover:text-primary/80">Complete</button>
                  </td>
                </tr>
              ))
            )}
            {!isLoading && tasks?.length === 0 && (
              <tr><td colSpan={5} className="py-12 text-center text-gray-500 text-sm">No housekeeping tasks found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export const MaintenancePage = () => {
  const { data: tickets, isLoading } = useQuery({
    queryKey: ['maintenance'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('maintenance_tickets')
        .select('*, units(name, properties(name))')
        .order('created_at', { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Maintenance</h1>
          <p className="text-sm text-gray-500">Track and resolve facility issues</p>
        </div>
        <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary hover:bg-primary/90 shadow-sm">
          <Plus className="w-4 h-4 mr-2" />
          New Ticket
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {isLoading ? (
          <div>Loading tickets...</div>
        ) : (
          tickets?.map((ticket: any) => (
            <div key={ticket.id} className="bg-white p-5 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-base font-semibold text-gray-900">{ticket.issue}</h3>
                  <div className="text-xs text-gray-500">{ticket.units?.properties?.name} - {ticket.units?.name}</div>
                </div>
                <span className={`px-2 py-1 text-[10px] font-bold rounded-md uppercase ${
                  ticket.priority === 'urgent' ? 'bg-red-100 text-red-700' : 'bg-orange-100 text-orange-700'
                }`}>
                  {ticket.priority}
                </span>
              </div>
              <div className="flex items-center justify-between pt-4 border-t border-gray-50">
                <div className="flex items-center text-xs text-gray-500">
                  <Clock className="w-3 h-3 mr-1" />
                  Opened {new Date(ticket.created_at).toLocaleDateString()}
                </div>
                <button className="text-xs font-medium text-primary hover:text-primary/80">Update Status</button>
              </div>
            </div>
          ))
        )}
        {!isLoading && tickets?.length === 0 && (
          <div className="col-span-full py-12 text-center bg-white rounded-xl border-2 border-dashed border-gray-200">
            <AlertTriangle className="mx-auto h-12 w-12 text-gray-300 mb-2" />
            <div className="text-gray-500">No active maintenance tickets.</div>
          </div>
        )}
      </div>
    </div>
  );
};
