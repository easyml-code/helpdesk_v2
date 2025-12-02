SYSTEM_PROMPT_HELPDESK="""
You are a helpful AI assistant with access to invoice database.
When users ask about invoices or vendors or anything related to them, use the "get_invoice_details" tool.
Provide clear and accurate responses.
"""

DATABASE_SCHEMA="""
CREATE TABLE public.invoices (
  invoice_number text NOT NULL,
  vendor_id bigint,
  customer text,
  invoice_date date,
  payment_terms text,
  due_date date,
  po_number text,
  item_description text,
  spend_category text,
  quantity bigint,
  item_invoice_rate double precision,
  tax_rate double precision,
  amount double precision,
  balance double precision,
  invoice_total double precision,
  payment_instruction text,
  status text,
  id uuid DEFAULT gen_random_uuid(),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT invoices_pkey PRIMARY KEY (invoice_number),
  CONSTRAINT invoices_vendor_id_fkey FOREIGN KEY (vendor_id) REFERENCES public.vendors(vendor_id)
);
CREATE TABLE public.vendor_users (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  vendor_id bigint NOT NULL,
  role text DEFAULT 'viewer'::text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT vendor_users_pkey PRIMARY KEY (id)
);
CREATE TABLE public.vendors (
  vendor_id bigint NOT NULL,
  vendor_name text,
  address text,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT vendors_pkey PRIMARY KEY (vendor_id)
);
"""