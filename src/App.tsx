import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuth } from './hooks/useAuth';
import { ProtectedRoute } from './components/auth/ProtectedRoute';
import { DashboardLayout } from './components/layout/DashboardLayout';
import { LoginPage } from './pages/auth/LoginPage';
import { PropertiesPage } from './pages/properties/PropertiesPage';
import { UnitsPage } from './pages/units/UnitsPage';
import { BookingsPage } from './pages/bookings/BookingsPage';
import { OccupancyPage } from './pages/occupancy/OccupancyPage';
import { GuestsPage } from './pages/guests/GuestsPage';
import { FinancePage } from './pages/finance/FinancePage';
import { HousekeepingPage, MaintenancePage } from './pages/operations/OperationsPage';
import { PublicPropertyPage } from './pages/public/PublicPropertyPage';
import PortfolioPage from './portfolio/PortfolioPage';

// Placeholder components for routes
const Dashboard = () => <div>Dashboard Overview</div>;
const Settings = () => <div>Settings</div>;

const queryClient = new QueryClient();

function App() {
  useAuth();

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Portfolio as the landing page */}
          <Route path="/" element={<PortfolioPage />} />

          <Route path="/login" element={<LoginPage />} />
          <Route path="/p/:slug" element={<PublicPropertyPage />} />

          <Route element={<ProtectedRoute><DashboardLayout /></ProtectedRoute>}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/properties" element={<PropertiesPage />} />
            <Route path="/units" element={<UnitsPage />} />
            <Route path="/bookings" element={<BookingsPage />} />
            <Route path="/occupancy" element={<OccupancyPage />} />
            <Route path="/guests" element={<GuestsPage />} />
            <Route path="/finance" element={<FinancePage />} />
            <Route path="/housekeeping" element={<HousekeepingPage />} />
            <Route path="/maintenance" element={<MaintenancePage />} />
            <Route path="/settings" element={<Settings />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
