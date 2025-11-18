# Docker 개발 환경 가이드

이 폴더에서 Docker를 사용하여 개별 개발 환경을 구성할 수 있습니다.

## 🚀 빠른 시작

### 1. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 설정하세요:

```bash
# 파트 이름 (ye, yh, js, sh 중 하나)
PART_NAME=your-part

# 포트 번호 (파트별로 다름)
# ye: 8010, yh: 8011, js: 8012, sh: 8013
PORT=8000

# PostgreSQL 설정
DB_NAME=feedlyai
DB_USER=feedlyai
DB_PASSWORD=feedlyai_dev_password_74154
DB_PORT=5434  # 파트별로 다름 (ye: 5434, yh: 5435, js: 5436, sh: 5437)

# Assets 디렉토리
ASSETS_DIR=/opt/feedlyai/assets

# 스키마 디렉토리 (팀 도커의 스키마 사용)
SCHEMA_DIR=../feedlyai/db/init

# Adminer 포트 (파트별로 다름)
# ye: 8083, yh: 8084, js: 8085, sh: 8086
ADMINER_PORT=8083
```

### 2. Docker Compose로 실행

```bash
# 빌드 및 실행
docker compose up --build

# 백그라운드 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f app

# 중지
docker compose down
```

## 📋 파트별 포트 설정

| 파트 | 서비스 포트 | PostgreSQL 포트 | Adminer 포트 |
|------|------------|----------------|--------------|
| ye   | 8010       | 5434           | 8083         |
| yh   | 8011       | 5435           | 8084         |
| js   | 8012       | 5436           | 8085         |
| sh   | 8013       | 5437           | 8086         |

## 🔧 설정 옵션

### 개별 PostgreSQL 사용 (기본)

현재 설정은 개별 PostgreSQL을 사용합니다.
- 각 파트별로 독립적인 데이터베이스 볼륨 사용
- 포트 충돌 방지


## 📁 볼륨 마운트

- **코드**: 현재 디렉토리(`.`) → `/app` (코드 변경 시 자동 반영)
- **Assets**: `/opt/feedlyai/assets` → `/assets` (팀 도커와 동일)

## 🧪 테스트

```bash
# Health check
curl http://localhost:${PORT}/healthz

# API 문서 확인
# 브라우저에서 http://localhost:${PORT}/docs 접속
```

## 🔍 유용한 명령어

```bash
# 컨테이너 상태 확인
docker compose ps

# 특정 서비스 재시작
docker compose restart app

# 컨테이너 내부 접속
docker compose exec app bash

# 로그 확인
docker compose logs app
docker compose logs postgres

# 볼륨 확인
docker volume ls

# 완전히 정리 (볼륨 포함)
docker compose down -v
```

## 📝 주의사항

1. **포트 충돌**: 다른 팀원과 포트가 겹치지 않도록 `.env`에서 설정 확인
2. **Assets 디렉토리**: 팀 도커와 동일한 경로(`/opt/feedlyai/assets`) 사용
3. **코드 변경**: 볼륨 마운트로 코드 변경 시 자동 반영 (--reload 옵션)
4. **데이터베이스**: 개별 postgres를 사용하면 데이터는 독립적으로 관리됨


## 문제 해결

### 포트 충돌
`.env` 파일에서 포트를 변경하세요.

### Assets 디렉토리 접근 불가
```bash
# 권한 확인
ls -la /opt/feedlyai/assets/

# 필요시 권한 설정 (팀장에게 요청)
```

### 데이터베이스 연결 오류
```bash
# postgres 컨테이너 상태 확인
docker compose ps postgres

# 로그 확인
docker compose logs postgres
```

