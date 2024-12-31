import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    APP_NAME: str = os.getenv('APP_NAME')
    DEBUG: bool = os.getenv('DEBUG')
    JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY')
    ALGORITHM: str = os.getenv('ALGORITHM')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES')
    REFRESH_TOKEN_EXPIRE_MINUTES: int = os.getenv('REFRESH_TOKEN_EXPIRE_MINUTES')
    REDIS_HOST: str = os.getenv('REDIS_HOST')
    REDIS_PORT: int = os.getenv('REDIS_PORT')
    JWT_REFRESH_SECRET_KEY: str = os.getenv('JWT_REFRESH_SECRET_KEY')
    MAIL_USERNAME: str = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD: str = os.getenv('MAIL_PASSWORD')
    MAIL_PORT: int = os.getenv('MAIL_PORT')
    MAIL_DEBUG: bool = os.getenv('MAIL_DEBUG')
    MAIL_SERVER: str = os.getenv('MAIL_SERVER')
    MAIL_STARTTLS: bool = os.getenv('MAIL_STARTTLS')
    MAIL_SSL_TLS: bool = os.getenv('MAIL_SSL_TLS')
    MAIL_FROM: str = os.getenv('MAIL_FROM')
    MAIL_FROM_NAME: str = os.getenv('MAIL_FROM_NAME')
    TEMPLATE_FOLDER: str = Path(__file__).parent.parent / os.getenv('TEMPLATE_FOLDER')
    USE_CREDENTIALS: bool = os.getenv('USE_CREDENTIALS')
    API_V1_STR: str = os.getenv('API_V1_STR')
    MONGO_CONNECTION_STRING: str = os.getenv('MONGO_CONNECTION_STRING')
    MONGO_CONNECTION_STRING2: str = os.getenv('MONGO_CONNECTION_STRING2')
    MONGO_INITDB_ROOT_USERNAME: str = os.getenv('MONGO_INITDB_ROOT_USERNAME')
    MONGO_INITDB_ROOT_PASSWORD: str = os.getenv('MONGO_INITDB_ROOT_PASSWORD')
    SECRET_KEY: str = os.getenv('SECRET_KEY')
    WEAVIATE_HOST: str = os.getenv('WEAVIATE_HOST')
    RESET_TOKEN_EXPIRE_MINUTES: int = os.getenv('RESET_TOKEN_EXPIRE_MINUTES')
    SESSION_RESET_TOKEN_EXPIRE_MINUTES: int = os.getenv('SESSION_RESET_TOKEN_EXPIRE_MINUTES')
    BACKEND_PORT: int = os.getenv('BACKEND_PORT')
    FRONTEND_PORT: int = os.getenv('FRONTEND_PORT')
    MINIO_ACCESS_KEY: str = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_ACCESS_KEY: str = os.getenv('MINIO_SECRET_ACCESS_KEY')
    BUCKET_NAME: str = os.getenv('BUCKET_NAME')
    REGION_NAME: str = os.getenv('REGION_NAME')
    HUGGINGFACE_API_KEY: str = os.getenv('HUGGINGFACE_API_KEY')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    COHERE_API_KEY: str = os.getenv('COHERE_API_KEY')
    SERVER_IP: str = os.getenv('SERVER_IP')
    MINIO_PORT: int = os.getenv('MINIO_PORT')
    WEAVIATE_API_KEY: str = os.getenv('WEAVIATE_API_KEY')
    WEAVIATE_CLUSTER_URL: str = os.getenv('WEAVIATE_CLUSTER_URL')
    MINIO_HOST: str = os.getenv('MINIO_HOST')
    REDIS_PASSWORD: str = os.getenv('REDIS_PASSWORD')
    MODEL_EMBEDDING_NAME: str = os.getenv('MODEL_EMBEDDING_NAME')
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY')
    MODEL_GENERATE_NAME: str = os.getenv('MODEL_GENERATE_NAME')
    NGROK_API_KEY: str = os.getenv('NGROK_API_KEY')
    FRONTEND_HOST: str = os.getenv('FRONTEND_HOST')
    class Config:
        env_file = env_path
        env_file_encoding = 'utf-8'


@lru_cache()
def get_settings() -> Settings:
    return Settings()
