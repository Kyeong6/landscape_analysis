import os

from dotenv import load_dotenv

# .env 참조: 클라우드에서 .env 이용
load_dotenv()

class Settings:
    
    CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH")

settings = Settings()