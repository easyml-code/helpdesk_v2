from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Supabase Config
    SUPABASE_URL: str
    SUPABASE_KEY: str
    ANON_KEY: str
    JWT_SECRET: str
    SUPABASE_HOST: str
    SUPABASE_DB: str = "postgres"
    POSTGRES_PASSWORD: str
    SUPABASE_PORT: int = 5432
    SUPABASE_USER: str = "postgres"
    
    # Auth Config
    VENDOR_EMAIL: str
    VENDOR_PASSWORD: str
    
    # LLM Config
    GROQ_API_KEY: str
    LLM_MODEL: str  # FIXED: Was LLM_MODEl
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 8000
    
    # Chat Session Config
    MAX_CONTEXT_MULTIPLIER: int = 10
    SESSION_TIMEOUT_MINUTES: int = 55
    AUTO_SAVE_INTERVAL_MINUTES: int = 5
    CHAT_HISTORY_LIMIT: int = 50
    MESSAGE_HISTORY_LIMIT: int = 100
    
    # Checkpointing Config
    CHECKPOINT_WINDOW_SIZE: int = 10
    MAX_WINDOW_TOKENS: int = 8000
    USE_ADAPTIVE_WINDOWING: bool = True
    
    # Cache Config
    CACHE_MESSAGE_VERSIONS: bool = True
    MAX_VERSION_HISTORY: int = 5
    
    # Metrics Config
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Logging Config
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_MAX_SIZE_MB: int = 10
    LOG_BACKUP_COUNT: int = 10

    class Config:
        env_file = ".env"
        extra = "allow"
        env_file_encoding = "utf-8"

settings = Settings()