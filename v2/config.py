"""Configuration management for V2 API.

Centralizes all environment variable access for better testability and maintainability.
"""

import os
from typing import Optional


class Config:
    """Application configuration loaded from environment variables."""

    # Gemini AI API
    @staticmethod
    def gemini_api_key() -> Optional[str]:
        """Get Gemini API key from environment."""
        return os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Supabase configuration
    @staticmethod
    def supabase_url() -> Optional[str]:
        """Get Supabase project URL from environment."""
        return os.environ.get("SUPABASE_URL")

    @staticmethod
    def supabase_service_role_key() -> Optional[str]:
        """Get Supabase service role key from environment."""
        return os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    @staticmethod
    def supabase_jwt_secret() -> Optional[str]:
        """Get Supabase JWT secret for token verification."""
        return os.environ.get("SUPABASE_JWT_SECRET")

    # Modal API
    @staticmethod
    def modal_api_key() -> Optional[str]:
        """Get Modal API key from environment."""
        return os.environ.get("MODAL_API_KEY")

    # Helper methods
    @staticmethod
    def is_configured() -> bool:
        """Check if all required configuration is present."""
        return all([
            Config.gemini_api_key(),
            Config.supabase_url(),
            Config.supabase_service_role_key(),
        ])

    @staticmethod
    def get_missing_config() -> list[str]:
        """Get list of missing required configuration keys."""
        missing = []
        if not Config.gemini_api_key():
            missing.append("GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY")
        if not Config.supabase_url():
            missing.append("SUPABASE_URL")
        if not Config.supabase_service_role_key():
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        return missing


# Singleton instance for easy access
config = Config()
