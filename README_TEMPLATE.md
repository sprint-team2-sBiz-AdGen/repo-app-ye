# FeedlyAI 개발 폴더 설정 가이드

이 폴더는 FeedlyAI 팀 프로젝트의 개인 개발 환경입니다.

## 사전 요구사항

### uv 설치

이 프로젝트는 `uv`를 사용하여 Python 패키지를 관리합니다. 먼저 `uv`가 설치되어 있는지 확인하세요:

```bash
# uv 설치 확인
which uv

# 설치되어 있지 않은 경우 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 pip를 통해 설치
pip install uv
```

설치 후 터미널을 재시작하거나 다음 명령어로 PATH를 업데이트하세요:

```bash
source $HOME/.cargo/env  # uv가 cargo를 통해 설치된 경우
```

## 빠른 시작

### 1. 폴더 복사 및 이름 변경

```bash
# 홈 디렉토리로 이동
cd ~

# 템플릿 폴더 복사
cp -r /home/leeyoungho/feedlyai/team-template ~/feedlyai-work

# 또는 직접 생성한 경우
mkdir -p ~/feedlyai-work
cd ~/feedlyai-work
```

### 2. uv로 가상환경 설정

```bash
cd ~/feedlyai-work

# Python 3.11로 가상환경 생성
uv venv --python 3.11

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
uv pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# 파트 이름 (ye, yh, js, sh 중 하나)
PART_NAME=your-part

# 포트 번호 (파트별로 다름)
# ye: 8010, yh: 8011, js: 8012, sh: 8013
PORT=8000

# 데이터베이스 설정
DB_NAME=feedlyai
DB_USER=feedlyai
DB_PASSWORD=feedlyai_dev_password_74154
DB_PORT=5434  # 파트별로 다름 (ye: 5434, yh: 5435, js: 5436, sh: 5437)

# Assets 디렉토리
ASSETS_DIR=/opt/feedlyai/assets

# Adminer 포트 (파트별로 다름)
# ye: 8083, yh: 8084, js: 8085, sh: 8086
ADMINER_PORT=8083
EOF

# .env 파일 수정
nano .env  # 또는 원하는 에디터 사용
```

`.env` 파일에서 다음을 수정:
- `PART_NAME`: 본인의 파트 이름 (ye, yh, js, sh 중 하나)
- `PORT`: 본인의 포트 번호 (ye: 8010, yh: 8011, js: 8012, sh: 8013)
- `DB_PORT`: 데이터베이스 포트 (ye: 5434, yh: 5435, js: 5436, sh: 5437)
- `DB_PASSWORD`: 팀 도커의 `.env` 파일에서 `DB_PASSWORD` 확인
- `DATABASE_URL`: (선택사항) Docker 사용 시 자동 구성됨
  - 로컬 개발 시: `postgresql://feedlyai:비밀번호@localhost:5432/feedlyai`
  - Docker 사용 시: `host.docker.internal`을 통해 팀 도커의 postgres에 연결

### 4. 데이터베이스 연결 정보 확인

```bash
cd /home/leeyoungho/feedlyai

# DB 비밀번호 확인
cat .env | grep DB_PASSWORD
```

본인의 `.env` 파일에 `DB_PASSWORD`를 위에서 확인한 값으로 업데이트하세요.

### 5. 로컬 개발 서버 실행

```bash
cd ~/feedlyai-work
source .venv/bin/activate

# 서버 실행
python main.py
```

브라우저에서 확인:
- http://localhost:8000/healthz
- http://localhost:8000/docs (API 문서)

### 6. Docker로 실행

#### 옵션 A: 개별 Docker 환경 (권장)

개별 Docker 환경은 팀 도커의 PostgreSQL에 연결됩니다 (`host.docker.internal` 사용).

```bash
cd ~/feedlyai-work

# Docker Compose로 실행
docker compose up --build

# 백그라운드 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f app

# 중지
docker compose down
```

**주의**: 개별 Docker 환경은 팀 도커의 PostgreSQL에 연결되므로, 팀 도커의 postgres가 실행 중이어야 합니다.


## 폴더 구조

```
feedlyai-work/
├── main.py              # 메인 애플리케이션 파일
├── requirements.txt      # Python 의존성
├── pyproject.toml        # 프로젝트 설정 (uv 사용)
├── Dockerfile           # Docker 이미지 빌드 파일
├── docker-compose.yml   # Docker Compose 설정
├── .dockerignore        # Docker 빌드 제외 파일
├── .env                  # 환경 변수 (직접 생성, Git에 커밋하지 않음)
├── .gitignore           # Git 무시 파일
├── .venv/               # 가상환경 (자동 생성)
├── README.md            # 이 파일
└── README_DOCKER.md     # Docker 사용 가이드
```


## 개발 가이드

### API 엔드포인트 추가

`main.py`에 엔드포인트를 추가하세요:

```python
@app.post("/api/{PART_NAME}/your-endpoint")
def your_endpoint():
    return {"message": "Hello"}
```

### Assets 저장

```python
from PIL import Image

# 이미지 저장 예시
def save_image(image: Image.Image, tenant_id: str, kind: str):
    today = datetime.datetime.utcnow()
    rel_dir = f"{PART_NAME}/tenants/{tenant_id}/{kind}/{today.year}/{today.month:02d}/{today.day:02d}"
    abs_dir = os.path.join(ASSETS_DIR, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    # 저장 로직...
```

## 유용한 명령어

### 로컬 개발

```bash
# 가상환경 활성화
source .venv/bin/activate

# 서버 실행
python main.py

# 의존성 추가
uv pip install <package-name>
```

### Docker 사용

#### 개별 Docker 환경 (권장)

```bash
# 현재 작업 디렉토리에서
cd ~/feedlyai-work

# 서비스 시작
docker compose up -d --build

# 서비스 중지
docker compose down

# 서비스 재시작
docker compose restart app

# 로그 확인
docker compose logs -f app

# 컨테이너 상태 확인
docker compose ps
```

자세한 내용은 `README_DOCKER.md` 참고

#### 팀 도커 사용

```bash
# 서비스 시작
cd /home/leeyoungho/feedlyai
sg docker -c "docker-compose up -d app-your-part"

# 서비스 중지
sg docker -c "docker-compose stop app-your-part"

# 서비스 재시작
sg docker -c "docker-compose restart app-your-part"

# 로그 확인
sg docker -c "docker-compose logs -f app-your-part"
```

## 파트별 설정 정보

| 파트 | 서비스 포트 | PostgreSQL 포트 | Adminer 포트 | PART_NAME |
|------|------------|----------------|-------------|-----------|
| ye   | 8010       | 5434           | 8083        | ye        |
| yh   | 8011       | 5435           | 8084        | yh        |
| js   | 8012       | 5436           | 8085        | js        |
| sh   | 8013       | 5437           | 8086        | sh        |

## 문제 해결

### 포트 충돌

다른 팀원과 같은 포트를 사용하면 충돌이 발생합니다. `.env`에서 포트를 변경하세요. 위 표를 참고하여 본인 파트에 맞는 포트를 사용하세요.

### Assets 디렉토리 접근 불가

```bash
# 권한 확인
ls -la /opt/feedlyai/assets/

# 필요시 권한 설정 (팀장에게 요청)
```

### Docker 연결 오류

#### 개별 Docker 환경 사용 시

```bash
# 컨테이너 상태 확인
docker compose ps

# 로그 확인
docker compose logs app

# 팀 도커의 postgres가 실행 중인지 확인
cd /home/leeyoungho/feedlyai
sg docker -c "docker-compose ps postgres"
```

#### 팀 도커 사용 시

```bash
# 컨테이너 상태 확인
sg docker -c "docker-compose ps"

# 로그 확인
sg docker -c "docker-compose logs app-your-part"
```

## 문의

문제가 발생하면 팀장에게 문의하세요.

