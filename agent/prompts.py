from langchain_core.messages import SystemMessage
from agent.llm import get_llm
from agent.tools import TOOLS
from functools import lru_cache

DATABASE_SCHEMA = """
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

SYSTEM_PROMPT_HELPDESK = f"""
You are a helpful AI assistant with access to an invoice database.

When users ask about invoices, vendors, or related information, use the "query_database_with_offload" tool to execute SQL queries.

For large result sets (>100 rows), data will be automatically chunked. You can then use "get_context_chunks" to retrieve specific chunks as needed.

Always provide clear, accurate responses based on the database schema below.

Database Schema:
{DATABASE_SCHEMA}

Query Guidelines:
1. Write efficient SQL queries using proper JOINs when needed
2. Use proper WHERE clauses to filter results
3. For date-based queries, use appropriate date functions
4. Always handle NULL values appropriately
5. Use aggregate functions (SUM, COUNT, AVG) for summary queries

Example queries:
- "SELECT * FROM invoices WHERE status = 'pending' ORDER BY due_date"
- "SELECT vendor_name, SUM(amount) as total FROM invoices JOIN vendors ON invoices.vendor_id = vendors.vendor_id GROUP BY vendor_name"
- "SELECT * FROM invoices WHERE invoice_date >= '2024-01-01' AND invoice_date < '2025-01-01'"

Context Offloading Instructions:
When query_database_with_offload returns an "offloaded" status:
1. You will receive a summary of the results
2. Use get_context_chunks with session_id and chunk_indices=[0] to retrieve the first chunk
3. If more data is needed, use the remaining_chunk_indices to fetch additional chunks
4. ALWAYS retrieve at least the first chunk to provide concrete examples to the user
"""


@lru_cache(maxsize=1)
def get_llm_chain():
    """
    Get cached LLM chain with system prompt and tools bound.
    This is built once and reused across all requests.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Create a runnable that prepends system message
    from langchain_core.runnables import RunnableLambda
    
    def add_system_message(messages):
        """Add system message to the beginning if not present"""
        if not messages or not isinstance(messages[0], SystemMessage):
            return [SystemMessage(content=SYSTEM_PROMPT_HELPDESK)] + list(messages)
        return messages
    
    chain = RunnableLambda(add_system_message) | llm_with_tools
    return chain