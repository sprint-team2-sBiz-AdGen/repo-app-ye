import os
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# 환경 변수 설정
ASSETS_DIR = os.getenv("ASSETS_DIR", "/var/www/assets")
PART_NAME = os.getenv("PART_NAME", "your-part")  # ye, yh, js, sh 중 하나로 변경
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")

# Docker 환경에서는 'postgres' 호스트명 사용, 로컬에서는 'localhost'
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "feedlyai")
DB_USER = os.getenv("DB_USER", "feedlyai")
DB_PASSWORD = os.getenv("DB_PASSWORD", "feedlyai_dev_password_74154")

# DATABASE_URL이 명시적으로 설정되지 않았으면 구성
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app = FastAPI(title=f"app-{PART_NAME}")

# 파트별 assets 디렉토리
PART_ASSETS_DIR = os.path.join(ASSETS_DIR, PART_NAME)

# 데이터베이스 연결
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/healthz")
def health(db: Session = Depends(get_db)):
    # DB 연결 테스트
    db.execute(text("SELECT 1"))
    return {"ok": True, "service": f"app-{PART_NAME}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

