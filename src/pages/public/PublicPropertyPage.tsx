import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { supabase } from '../../lib/supabase';
import { MapPin, Star, Wifi, Coffee, Utensils, Shield, Calendar, Users } from 'lucide-react';

export const PublicPropertyPage = () => {
  const { slug } = useParams();

  const { data: property, isLoading: isPropertyLoading } = useQuery({
    queryKey: ['public-property', slug],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('properties')
        .select('*, unit_categories(*)')
        .eq('slug', slug)
        .single();
      if (error) throw error;
      return data;
    },
  });

  if (isPropertyLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!property) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white text-center p-8">
        <div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Property Not Found</h1>
          <p className="text-gray-500 mb-8">We couldn't find the resort you're looking for.</p>
          <a href="/" className="inline-flex items-center px-6 py-3 bg-primary text-white font-medium rounded-lg">Return Home</a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="relative h-[60vh] bg-gray-900 overflow-hidden">
        <img
          src={property.logo_url || '/src/assets/hero.png'}
          alt={property.name}
          className="w-full h-full object-cover opacity-70"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full pb-16">
            <h1 className="text-4xl sm:text-6xl font-extrabold text-white mb-4 tracking-tight">{property.name}</h1>
            <div className="flex items-center text-white/90 space-x-4">
              <div className="flex items-center">
                <MapPin className="w-5 h-5 mr-1" />
                {property.city}, {property. province}
              </div>
              <div className="flex items-center">
                <Star className="w-5 h-5 mr-1 text-yellow-400 fill-yellow-400" />
                4.9 (120 Reviews)
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-12">
            <section>
              <h2 className="text-3xl font-bold text-gray-900 mb-6">About this Resort</h2>
              <p className="text-lg text-gray-600 leading-relaxed">
                {property.description || "Experience luxury and serenity at our resort. Nestled in the heart of nature, we offer the perfect escape for families, couples, and solo travelers. Our accommodations are designed for comfort and elegance, ensuring a memorable stay."}
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Popular Amenities</h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-6">
                <div className="flex items-center p-4 bg-white rounded-xl border border-gray-100"><Wifi className="w-5 h-5 mr-3 text-primary" /> Free WiFi</div>
                <div className="flex items-center p-4 bg-white rounded-xl border border-gray-100"><Coffee className="w-5 h-5 mr-3 text-primary" /> Breakfast</div>
                <div className="flex items-center p-4 bg-white rounded-xl border border-gray-100"><Utensils className="w-5 h-5 mr-3 text-primary" /> Restaurant</div>
                <div className="flex items-center p-4 bg-white rounded-xl border border-gray-100"><Shield className="w-5 h-5 mr-3 text-primary" /> 24/7 Security</div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Choose Your Stay</h2>
              <div className="space-y-6">
                {property.unit_categories?.map((category: any) => (
                  <div key={category.id} className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden flex flex-col sm:flex-row hover:shadow-md transition-shadow">
                    <div className="sm:w-64 h-48 bg-gray-200">
                      <div className="w-full h-full flex items-center justify-center text-gray-400 font-medium italic">Room Image</div>
                    </div>
                    <div className="p-6 flex-1 flex flex-col justify-between">
                      <div>
                        <div className="flex justify-between items-start mb-2">
                          <h3 className="text-xl font-bold text-gray-900">{category.name}</h3>
                          <span className="text-primary font-bold text-lg">Rs. {category.base_price?.toLocaleString()}</span>
                        </div>
                        <div className="flex items-center text-gray-500 text-sm mb-4">
                          <Users className="w-4 h-4 mr-1" />
                          Up to {category.capacity_adults} Adults
                        </div>
                      </div>
                      <button className="w-full sm:w-auto px-6 py-2 bg-primary text-white rounded-lg font-medium hover:bg-primary/90 transition-colors">
                        Reserve Now
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* Sticky Booking Card */}
          <div className="lg:col-span-1">
            <div className="sticky top-24 bg-white rounded-2xl shadow-xl border border-gray-100 p-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Book Your Escape</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-1">Check-in</label>
                  <div className="relative">
                    <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input type="date" className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:ring-primary focus:border-primary" />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-1">Check-out</label>
                  <div className="relative">
                    <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input type="date" className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-xl focus:ring-primary focus:border-primary" />
                  </div>
                </div>
                <div className="pt-4">
                  <button className="w-full py-4 bg-primary text-white rounded-xl font-bold text-lg shadow-lg shadow-primary/20 hover:scale-[1.02] active:scale-[0.98] transition-all">
                    Check Availability
                  </button>
                </div>
              </div>
              <p className="mt-4 text-center text-sm text-gray-500 italic">Secure booking powered by ResortERP</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
