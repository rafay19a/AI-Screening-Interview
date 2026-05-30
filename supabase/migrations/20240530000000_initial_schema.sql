-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Organizations
create table organizations (
  id uuid primary key default uuid_generate_v4(),
  name text not null,
  slug text unique not null,
  subscription_plan text default 'starter',
  subscription_status text default 'active',
  created_at timestamp with time zone default now()
);

-- Profiles (linked to Supabase Auth)
create table profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  organization_id uuid references organizations(id),
  full_name text,
  role text check (role in ('platform_admin', 'org_owner', 'resort_manager', 'sales_staff', 'accountant')),
  created_at timestamp with time zone default now()
);

-- Properties
create table properties (
  id uuid primary key default uuid_generate_v4(),
  organization_id uuid references organizations(id) not null,
  name text not null,
  description text,
  address text,
  city text,
  province text,
  country text,
  gps_coordinates text,
  contact_email text,
  contact_phone text,
  logo_url text,
  created_at timestamp with time zone default now()
);

-- Unit Categories
create table unit_categories (
  id uuid primary key default uuid_generate_v4(),
  property_id uuid references properties(id) not null,
  name text not null, -- e.g., 'Executive', 'Deluxe'
  base_price decimal(12,2) not null,
  capacity_adults int default 2,
  capacity_children int default 0,
  amenities text[],
  created_at timestamp with time zone default now()
);

-- Units
create table units (
  id uuid primary key default uuid_generate_v4(),
  property_id uuid references properties(id) not null,
  category_id uuid references unit_categories(id) not null,
  name text not null, -- e.g., 'Room 101', 'Cottage A'
  status text check (status in ('available', 'occupied', 'maintenance', 'dirty')) default 'available',
  created_at timestamp with time zone default now()
);

-- Guests
create table guests (
  id uuid primary key default uuid_generate_v4(),
  organization_id uuid references organizations(id) not null,
  first_name text not null,
  last_name text not null,
  email text,
  phone text,
  document_type text,
  document_number text,
  preferences text,
  is_vip boolean default false,
  created_at timestamp with time zone default now()
);

-- Bookings
create table bookings (
  id uuid primary key default uuid_generate_v4(),
  organization_id uuid references organizations(id) not null,
  property_id uuid references properties(id) not null,
  guest_id uuid references guests(id) not null,
  check_in date not null,
  check_out date not null,
  status text check (status in ('inquiry', 'tentative', 'confirmed', 'checked_in', 'checked_out', 'completed', 'cancelled', 'no_show')) default 'confirmed',
  source text,
  total_amount decimal(12,2),
  notes text,
  created_at timestamp with time zone default now()
);

-- Booking Units (Many-to-many between bookings and units)
create table booking_units (
  booking_id uuid references bookings(id) on delete cascade,
  unit_id uuid references units(id),
  primary key (booking_id, unit_id)
);

-- Payments
create table payments (
  id uuid primary key default uuid_generate_v4(),
  booking_id uuid references bookings(id) not null,
  amount decimal(12,2) not null,
  payment_method text check (payment_method in ('cash', 'bank_transfer', 'easypaisa', 'jazzcash', 'stripe', 'card')),
  reference_number text,
  status text check (status in ('pending', 'completed', 'failed', 'refunded')) default 'completed',
  created_at timestamp with time zone default now()
);

-- Housekeeping Tasks
create table housekeeping_tasks (
  id uuid primary key default uuid_generate_v4(),
  unit_id uuid references units(id) not null,
  assigned_to uuid references profiles(id),
  status text check (status in ('pending', 'in_progress', 'completed')) default 'pending',
  priority text default 'medium',
  notes text,
  created_at timestamp with time zone default now()
);

-- Maintenance Tickets
create table maintenance_tickets (
  id uuid primary key default uuid_generate_v4(),
  unit_id uuid references units(id) not null,
  issue text not null,
  priority text check (priority in ('low', 'medium', 'high', 'urgent')),
  status text check (status in ('open', 'in_progress', 'resolved', 'closed')) default 'open',
  assigned_to uuid references profiles(id),
  created_at timestamp with time zone default now()
);

-- Enable RLS
alter table organizations enable row level security;
alter table profiles enable row level security;
alter table properties enable row level security;
alter table unit_categories enable row level security;
alter table units enable row level security;
alter table guests enable row level security;
alter table bookings enable row level security;
alter table booking_units enable row level security;
alter table payments enable row level security;
alter table housekeeping_tasks enable row level security;
alter table maintenance_tickets enable row level security;

-- RLS Policies (Simplified for multi-tenancy)

-- Profiles
create policy "Users can see their own profile" on profiles
  for select using (auth.uid() = id);

create policy "Users can update their own profile" on profiles
  for update using (auth.uid() = id);

-- Organizations
create policy "Users can see their organization" on organizations
  for select using (id in (select organization_id from profiles where profiles.id = auth.uid()));

-- Shared policy template for organization-based tables
-- properties, guests, bookings
create policy "Org isolation for properties" on properties
  using (organization_id in (select organization_id from profiles where profiles.id = auth.uid()));

create policy "Org isolation for guests" on guests
  using (organization_id in (select organization_id from profiles where profiles.id = auth.uid()));

create policy "Org isolation for bookings" on bookings
  using (organization_id in (select organization_id from profiles where profiles.id = auth.uid()));

-- Tables linked via property_id or unit_id (indirect org check)
create policy "Org isolation for unit_categories" on unit_categories
  using (property_id in (select id from properties where organization_id in (select organization_id from profiles where profiles.id = auth.uid())));

create policy "Org isolation for units" on units
  using (property_id in (select id from properties where organization_id in (select organization_id from profiles where profiles.id = auth.uid())));

create policy "Org isolation for payments" on payments
  using (booking_id in (select id from bookings where organization_id in (select organization_id from profiles where profiles.id = auth.uid())));
