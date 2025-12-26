# app/core/config.py

# Import 'BaseSettings' and 'SettingsConfigDict' from the pydantic-settings library.
# These help us manage configuration variables and validate their types automatically.
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define a 'Settings' class that inherits from 'BaseSettings'.
# This class will define all the configuration variables for our application.
class Settings(BaseSettings):
    
    # Define the name of the project; the type is explicitly set to string.
    # Default value is "PatentAI Projesi".
    PROJECT_NAME: str = "PatentAI Projesi"
    
    # Define the URL prefix for the API versions (e.g., all endpoints will start with /api/v1).
    API_V1_STR: str = "/api/v1"

    # This special configuration tells Pydantic to read settings from a '.env' file.
    # This is useful for keeping secrets (like passwords) out of the code.
    model_config = SettingsConfigDict(env_file=".env")

# Create an instance of the Settings class.
# We will import this 'settings' object in other files to access our configuration.
settings = Settings()