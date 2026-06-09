import { useQuery } from '@tanstack/react-query';
import { supabase } from '../../lib/supabase';
import { Plus, User, Mail, Phone, Star } from 'lucide-react';

export const GuestsPage = () => {
  const { data: guests, isLoading } = useQuery({
    queryKey: ['guests'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('guests')
        .select('*')
        .order('last_name');
      if (error) throw error;
      return data;
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Guest CRM</h1>
          <p className="text-sm text-gray-500">Manage guest profiles, preferences, and stay history</p>
        </div>
        <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary hover:bg-primary/90 shadow-sm">
          <Plus className="w-4 h-4 mr-2" />
          Add Guest
        </button>
      </div>

      <div className="bg-white shadow-sm border border-gray-100 rounded-xl overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Guest Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Contact</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Documents</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {isLoading ? (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center">
                   <div className="flex justify-center"><div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div></div>
                </td>
              </tr>
            ) : (
              guests?.map((guest: any) => (
                <tr key={guest.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="h-10 w-10 rounded-full bg-primary/5 flex items-center justify-center mr-3 text-primary font-bold">
                        {guest.first_name[0]}{guest.last_name[0]}
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-900 flex items-center">
                          {guest.first_name} {guest.last_name}
                          {guest.is_vip && <Star className="w-3 h-3 ml-2 text-yellow-400 fill-yellow-400" />}
                        </div>
                        <div className="text-xs text-gray-500">Joined {new Date(guest.created_at).toLocaleDateString()}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-600 flex items-center mb-1">
                      <Mail className="w-3 h-3 mr-2" /> {guest.email}
                    </div>
                    <div className="text-sm text-gray-600 flex items-center">
                      <Phone className="w-3 h-3 mr-2" /> {guest.phone}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {guest.is_vip ? <span className="text-primary font-medium">VIP</span> : 'Standard'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {guest.document_type || 'None'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <button className="text-primary hover:text-primary/80">View Profile</button>
                  </td>
                </tr>
              ))
            )}
            {!isLoading && guests?.length === 0 && (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center text-gray-500 text-sm">
                  <User className="mx-auto h-12 w-12 text-gray-300 mb-2" />
                  No guest profiles found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
