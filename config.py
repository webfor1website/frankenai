"""
FrankenAI v12.0 - Configuration
"""

import os
from typing import Optional


class FrankenAIConfig:
    """Configuration management"""
    
    def __init__(self):
        # API Keys
        self.groq_api_key: Optional[str] = None
        self.claude_api_key: Optional[str] = None
        self.grok_api_key: Optional[str] = None
        
        # Flask
        self.flask_secret_key: str = 'frankenai_secret_key_2026'
        
        # Storage
        self.storage_backend: str = 'file'
        self.storage_base_dir: str = 'cache'
        self.s3_bucket: Optional[str] = None
        self.s3_prefix: str = 'frankenai/cache/'
        
        # Caching
        self.cache_ttl: int = 3600
        
        # Conversations
        self.max_conversations: int = 1000
        self.conversation_ttl: int = 3600
        
        # Monitoring
        self.metrics_enabled: bool = True
        self.health_check_enabled: bool = True
        self.aws_secrets_manager_enabled: bool = False
        
        # Pricing (per million tokens)
        self.claude_input_per_m: float = 3.0
        self.claude_output_per_m: float = 15.0
        self.grok_input_per_m: float = 5.0
        self.grok_output_per_m: float = 15.0
    
    @classmethod
    def from_environment(cls) -> 'FrankenAIConfig':
        """Load config from environment variables"""
        config = cls()
        
        config.groq_api_key = os.getenv('GROQ_API_KEY')
        config.claude_api_key = os.getenv('CLAUDE_API_KEY')
        config.grok_api_key = os.getenv('GROK_API_KEY')
        config.flask_secret_key = os.getenv('FLASK_SECRET_KEY', config.flask_secret_key)
        config.storage_backend = os.getenv('STORAGE_BACKEND', 'file')
        config.storage_base_dir = os.getenv('STORAGE_BASE_DIR', 'cache')
        config.s3_bucket = os.getenv('S3_BUCKET')
        config.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))
        config.max_conversations = int(os.getenv('MAX_CONVERSATIONS', '1000'))
        config.conversation_ttl = int(os.getenv('CONVERSATION_TTL', '3600'))
        config.metrics_enabled = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        config.health_check_enabled = os.getenv('HEALTH_CHECK_ENABLED', 'true').lower() == 'true'
        
        return config
    
    def validate(self) -> None:
        """Validate required config"""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
    
    def to_dict(self) -> dict:
        """Return config as dict (hide sensitive values)"""
        return {
            'storage_backend': self.storage_backend,
            'cache_ttl': self.cache_ttl,
            'max_conversations': self.max_conversations,
            'metrics_enabled': self.metrics_enabled,
            'health_check_enabled': self.health_check_enabled,
            'groq_configured': bool(self.groq_api_key),
            'claude_configured': bool(self.claude_api_key),
            'grok_configured': bool(self.grok_api_key)
        }