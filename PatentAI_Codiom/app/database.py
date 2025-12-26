# app/database.py

# Import the 'create_engine' function to manage the connection to the database
from sqlalchemy import create_engine

# Import 'declarative_base' to create a base class that our database models will inherit from
from sqlalchemy.ext.declarative import declarative_base

# Import 'sessionmaker' to create a factory for database sessions
from sqlalchemy.orm import sessionmaker

# Define the database URL; this specifies we are using SQLite and the file is 'patentai.db'
SQLALCHEMY_DATABASE_URL = "sqlite:///./patentai.db"

# Create the SQLAlchemy engine, which acts as the core interface to the database
# Note: 'connect_args={"check_same_thread": False}' is necessary specifically for SQLite 
# to allow more than one thread to interact with the database at the same time
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a 'SessionLocal' class; this is a factory that produces new database sessions
# autocommit=False: Changes are not saved automatically; we must commit them manually
# autoflush=False: Changes are not pushed to the database automatically
# bind=engine: Connects these sessions to the engine created above
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a 'Base' class; all our database models (tables) will inherit from this class
Base = declarative_base()