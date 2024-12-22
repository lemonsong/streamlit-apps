import dotenv
import os

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

QUANDL_API = os.getenv("QUANDL_API")
FRED_API = os.getenv("FRED_API")
# DATABASE_DB= os.getenv("DATABASE_DB")
# DATABASE_USER= os.getenv("DATABASE_USER")
# DATABASE_PW= os.getenv("DATABASE_PW")
# DATABASE_HOST= os.getenv("DATABASE_HOST")
# DATABASE_PORT= os.getenv("DATABASE_PORT")
