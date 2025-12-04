import os

DB_HOST = os.getenv("DB_HOST", "localhost" if not os.path.exists("/.dockerenv") else "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "feedlyai")
DB_USER = os.getenv("DB_USER", "feedlyai")
DB_PASSWORD = os.getenv("DB_PASSWORD", "feedlyai_dev_password_74154")

# DATABASE_URL이 명시적으로 설정되지 않았거나 빈 문자열이면 자동 구성
DATABASE_URL = os.getenv("DATABASE_URL") or f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

ASSETS_DIR = os.getenv("ASSETS_DIR", "/opt/feedlyai/assets")
PART_NAME = os.getenv("PART_NAME", "ye")

# URL 루트 (/assets/ye/...)
ASSETS_URL_PREFIX = os.getenv("ASSETS_URL_PREFIX", "/assets")

# 테넌트 루트 (/opt/feedlyai/assets/ye/tenants)
TENANTS_DIR_NAME = os.getenv("TENANTS_DIR_NAME", "tenants")
GENERATED_DIR_NAME = os.getenv("GENERATED_DIR_NAME", "generated")

TENANTS_ROOT = os.path.join(ASSETS_DIR, PART_NAME, TENANTS_DIR_NAME)

ENABLE_JOB_STATE_LISTENER = True
JOB_STATE_LISTENER_RECONNECT_DELAY=5