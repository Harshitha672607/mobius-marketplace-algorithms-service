from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Valuation Engine API"
    app_version: str = "1.0.0"
    default_timeout: int = 10

    # Feature flags
    pi_layer_mode: str = "mock"   # future: "sql"

    class Config:
        env_file = ".env"

settings = Settings()
