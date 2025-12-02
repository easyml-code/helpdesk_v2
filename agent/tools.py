from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Any
from logs.log import logger

@tool
def get_invoice_details(query: str) -> List[Dict[str, Any]]:
    """Fetch invoice details from the database based on user query
    
    Args:
        query: PostgreSQL Select query or search parameters for invoice details
    
    Returns:
        List of invoice records
    """
    try:
        from langchain_core.runnables import RunnablePassthrough
        from database.client import run_query
        
        # Get config from context (passed automatically by ToolNode)
        config = RunnablePassthrough.get_config()
        access_token = config.get("configurable", {}).get("access_token")
        
        results = run_query(query=query, access_token=access_token)
        logger.info(f"✅ Fetched {len(results)} invoice records")
        return results
    except Exception as e:
        logger.error(f"❌ Error fetching invoices: {e}", exc_info=True)
        return []

# List of all tools
TOOLS = [get_invoice_details]