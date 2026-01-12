from langchain_groq import ChatGroq
from config import settings
from typing import Optional
from logs.log import logger

class LLMClient:
    """LLM client wrapper for Groq"""
    
    def __init__(self):
        self._llm: Optional[ChatGroq] = None
    
    def get_llm(self) -> ChatGroq:
        """Get or create Groq LLM instance"""
        if self._llm is None:
            try:
                self._llm = ChatGroq(
                    model=settings.LLM_MODEL,  # FIXED: Was LLM_MODEl
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    timeout=60.0,
                    max_retries=3,
                )
                logger.info(f"llm_initialized - model={settings.LLM_MODEL}, provider=groq")
            except Exception as e:
                logger.error(f"llm_initialization_failed - error={e}", exc_info=True)
                raise
        
        return self._llm
    
    def reset_llm(self):
        """Reset LLM instance (useful for testing or configuration changes)"""
        self._llm = None
        logger.info("llm_instance_reset")


# Global instance
llm_client = LLMClient()


def get_llm() -> ChatGroq:
    """Dependency for tools and agents"""
    return llm_client.get_llm()