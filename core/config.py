import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    AFTER = os.getenv("AFTER")
    AFTER_AFRICA = os.getenv("AFTER_AFRICA")
    AFTER_ANTARCTICA = os.getenv("AFTER_ANTARCTICA")
    AFTER_ASIA = os.getenv("AFTER_ASIA")
    AFTER_EUROPE = os.getenv("AFTER_EUROPE")
    AFTER_NORTH_AMERICA = os.getenv("AFTER_NORTH_AMERICA")
    AFTER_OCEANIA = os.getenv("AFTER_OCEANIA")
    AFTER_SOUTH_AMERICA = os.getenv("AFTER_SOUTH_AMERICA")
    BEFORE = os.getenv("BEFORE")
    BEFORE_AFRICA = os.getenv("BEFORE_AFRICA")
    BEFORE_ANTARCTICA = os.getenv("BEFORE_ANTARCTICA")
    BEFORE_ASIA = os.getenv("BEFORE_ASIA")
    BEFORE_EUROPE = os.getenv("BEFORE_EUROPE")
    BEFORE_NORTH_AMERICA = os.getenv("BEFORE_NORTH_AMERICA")
    BEFORE_OCEANIA = os.getenv("BEFORE_OCEANIA")
    BEFORE_SOUTH_AMERICA = os.getenv("BEFORE_SOUTH_AMERICA")
    
settings = Settings()
