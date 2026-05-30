import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { supabase } from '../../lib/supabase';
import { Plus, Building2, MapPin, Phone, Mail } from 'lucide-react';

export const PropertiesPage = () => {
  const [, setIsAddModalOpen] = useState(false);

  const { data: properties, isLoading } = useQuery({
    queryKey: ['properties'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('properties')
        .select('*')
        .order('name');
      if (error) throw error;
      return data;
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Properties</h1>
          <p className="text-sm text-gray-500">Manage your resort locations and properties</p>
        </div>
        <button
          onClick={() => setIsAddModalOpen(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary hover:bg-primary/90 shadow-sm"
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Property
        </button>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {properties?.map((property) => (
            <div key={property.id} className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow">
              <div className="h-32 bg-gray-100 flex items-center justify-center">
                {property.logo_url ? (
                  <img src={property.logo_url} alt={property.name} className="h-full w-full object-cover" />
                ) : (
                  <Building2 className="w-12 h-12 text-gray-300" />
                )}
              </div>
              <div className="p-5 space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">{property.name}</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <div className="flex items-center">
                    <MapPin className="w-4 h-4 mr-2 text-gray-400" />
                    {property.city}, {property.province}
                  </div>
                  <div className="flex items-center">
                    <Phone className="w-4 h-4 mr-2 text-gray-400" />
                    {property.contact_phone}
                  </div>
                  <div className="flex items-center">
                    <Mail className="w-4 h-4 mr-2 text-gray-400" />
                    {property.contact_email}
                  </div>
                </div>
                <div className="pt-4 border-t border-gray-50 flex justify-between">
                  <button className="text-sm font-medium text-primary hover:text-primary/80">View Details</button>
                  <button className="text-sm font-medium text-gray-500 hover:text-gray-700">Edit</button>
                </div>
              </div>
            </div>
          ))}
          {properties?.length === 0 && (
            <div className="col-span-full py-12 text-center bg-white rounded-xl border-2 border-dashed border-gray-200">
              <Building2 className="mx-auto h-12 w-12 text-gray-300" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No properties</h3>
              <p className="mt-1 text-sm text-gray-500">Get started by adding your first property.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
