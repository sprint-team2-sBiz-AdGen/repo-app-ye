# 팀원 개발 환경 설정 가이드

## 📋 설정 체크리스트

### 1단계: 폴더 준비

- [ ] 홈 디렉토리에 `feedlyai-work` 폴더 생성
- [ ] 템플릿 파일들을 복사

```bash
cd ~
cp -r /home/leeyoungho/feedlyai/team-template ~/feedlyai-work
cd ~/feedlyai-work
```

### 2단계: uv 가상환경 설정

- [ ] uv 설치 확인 (`which uv`)
- [ ] 가상환경 생성 (`uv venv --python 3.11`)
- [ ] 가상환경 활성화 (`source .venv/bin/activate`)
- [ ] 의존성 설치 (`uv pip install -r requirements.txt`)

### 3단계: 환경 변수 설정

- [ ] `.env` 파일 생성 (`cp .env.example .env`)
- [ ] `PART_NAME` 설정 (ye/js/sh 중 본인 파트)
- [ ] `PORT` 설정 (8010/8012/8013 중 본인 포트)
- [ ] `DATABASE_URL` 설정 (필수)
  - [ ] 팀 도커의 `.env`에서 `DB_PASSWORD` 확인
  - [ ] `DATABASE_URL=postgresql://feedlyai:비밀번호@postgres:5432/feedlyai` 형식으로 설정

### 4단계: 팀 도커 설정

- [ ] `/home/leeyoungho/feedlyai/.env` 파일 수정
- [ ] 본인의 `CODE_DIR` 경로 추가 확인

예시:
```bash
# ye 팀원인 경우
YE_CODE_DIR=/home/ye/feedlyai-work
```

### 5단계: 로컬 테스트

- [ ] 서버 실행 (`python main.py`)
- [ ] http://localhost:8000/healthz 접속 확인
- [ ] http://localhost:8000/docs 접속 확인

### 6단계: Docker 테스트

#### 옵션 A: 개별 Docker 환경 (권장)

- [ ] 작업 디렉토리에서 `.env` 파일 생성 (`.env.example` 참고)
- [ ] `PART_NAME`, `PORT`, `DB_PORT` 등 설정
- [ ] Docker Compose 실행 (`docker compose up -d --build`)
- [ ] 로그 확인 (`docker compose logs -f app`)
- [ ] Health check (`curl http://localhost:${PORT}/healthz`)

자세한 내용은 `README_DOCKER.md` 참고

#### 옵션 B: 팀 도커 사용

- [ ] 팀 도커 위치로 이동 (`cd /home/leeyoungho/feedlyai`)
- [ ] 본인 파트 실행 (`docker-compose up -d app-your-part`)
- [ ] 로그 확인 (`docker-compose logs -f app-your-part`)

## 🎯 각 파트별 설정

### app-ye (이미지 생성/분석)
- PORT: 8010
- PART_NAME: ye
- CODE_DIR: /home/ye/feedlyai-work
- DB_PORT: 5434 (개별 Docker 사용 시)
- ADMINER_PORT: 8083 (개별 Docker 사용 시)

### app-yh (YOLO/Planner/Overlay/Eval/Judge)
- PORT: 8011
- PART_NAME: yh
- CODE_DIR: /home/yh/feedlyai-work
- DB_PORT: 5435 (개별 Docker 사용 시)
- ADMINER_PORT: 8084 (개별 Docker 사용 시)

### app-js (FE/BFF & 업로드)
- PORT: 8012
- PART_NAME: js
- CODE_DIR: /home/js/feedlyai-work
- DB_PORT: 5436 (개별 Docker 사용 시)
- ADMINER_PORT: 8085 (개별 Docker 사용 시)

### app-sh (이미지 향상/배경 제거)
- PORT: 8013
- PART_NAME: sh
- CODE_DIR: /home/sh/feedlyai-work
- DB_PORT: 5437 (개별 Docker 사용 시)
- ADMINER_PORT: 8086 (개별 Docker 사용 시)

## ⚠️ 주의사항

1. **포트 충돌**: 다른 팀원과 같은 포트 사용 금지
2. **폴더 경로**: 각자 홈 디렉토리에 `feedlyai-work` 생성
3. **환경 변수**: `.env` 파일을 Git에 커밋하지 마세요
4. **Assets 권한**: `/opt/feedlyai/assets`는 공용 디렉토리입니다

## 🆘 문제 발생 시

1. 로그 확인: `docker-compose logs app-your-part`
2. 컨테이너 상태: `docker-compose ps`
3. 팀장에게 문의

