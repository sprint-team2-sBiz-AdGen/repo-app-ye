"""데이터베이스 모델 및 세션 관리"""

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Integer,
    Float,
    ForeignKey,
    TIMESTAMP,
    func,
    text,
    Boolean,
)
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    relationship,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.config import DATABASE_URL

# ======================================
#  Base / Engine / Session
# ======================================

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ======================================
#  TEST_ASSETS (테스트용 샘플 테이블)
# ======================================

class TestAsset(Base):
    """간단한 insert/delete 테스트용 Asset 테이블."""
    __tablename__ = "test_assets"

    image_asset_id = Column(UUID(as_uuid=True), primary_key=True)
    image_type = Column(String(50), nullable=True)
    image_url = Column(Text, nullable=True)
    mask_url = Column(Text, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    creator_id = Column(UUID(as_uuid=True), nullable=True)
    tenant_id = Column(String(255), nullable=True)
    uid = Column(String(255), nullable=True)
    pk = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


# ======================================
#  IMAGE_ASSETS
#   - 업로드 이미지 / 캔버스 / 마스크 / 결과 이미지 공통 저장소
# ======================================

class ImageAsset(Base):
    """이미지 파일 경로 및 메타데이터를 저장하는 테이블."""
    __tablename__ = "image_assets"

    image_asset_id = Column(UUID(as_uuid=True), primary_key=True)
    image_type = Column(String(50), nullable=True)
    image_url = Column(Text, nullable=True)
    mask_url = Column(Text, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    creator_id = Column(UUID(as_uuid=True), nullable=True)
    tenant_id = Column(String(255), nullable=True)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("jobs.job_id"),
        nullable=True,
        index=True,
    )
    pk = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


# ======================================
#  PBG_PROMPT_ASSETS
#   - 스타일별 기본/네거티브 프롬프트
# ======================================

class PbgPromptAsset(Base):
    """PBG 배경용 프롬프트 템플릿(pbg_prompt_assets)."""
    __tablename__ = "pbg_prompt_assets"

    prompt_asset_id = Column(UUID(as_uuid=True), primary_key=True)
    tone_style_id = Column(UUID(as_uuid=True), nullable=True)
    prompt_type = Column(Text, nullable=True)        # 스타일 이름 (예: "Hero Dish Focus")
    prompt_version = Column(Text, nullable=True)     # 프롬프트 버전 (예: "v1")
    prompt = Column(JSONB, nullable=True)            # {"en": "..."} 형태의 배경 프롬프트
    negative_prompt = Column(JSONB, nullable=True)   # {"en": "..."} 네거티브 프롬프트
    pk = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


# ======================================
#  PBG_PLACEMENT_PRESETS
#   - 피사체 위치/사이즈 프리셋
# ======================================

class PbgPlacementPreset(Base):
    """
    pbg_placement_presets 테이블.

    prompt_type 별로 variant1/2/3 에 대응하는 위치/사이즈 프리셋을 정의한다.
    """
    __tablename__ = "pbg_placement_presets"

    placement_preset_id = Column(UUID(as_uuid=True), primary_key=True)
    prompt_type = Column(Text, nullable=True)
    preset_order = Column(Integer, nullable=True)

    # 위치/사이즈 (일반적으로 0.0 ~ 1.0 범위로 저장)
    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    size = Column(Float, nullable=True)
    rotation = Column(Float, nullable=True)

    pk = Column(Integer)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=True,
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )


# =========================================================
#  GEN_MODELS
#   - 사용한 생성 모델 메타 정보
# =========================================================

class GenModel(Base):
    """생성 파이프라인에서 사용하는 모델 정보(gen_models)."""
    __tablename__ = "gen_models"

    model_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    name = Column(Text, nullable=True)
    repo = Column(Text, nullable=True)
    version = Column(Text, nullable=True)
    defaults = Column(JSONB, nullable=True)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=True,
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

    # 이 모델을 사용한 run 목록
    runs = relationship("GenRun", back_populates="model")


# =========================================================
#  GEN_RUNS
#   - 한 번의 "생성 작업(run)" 단위
#   - 원본/누끼 이미지, 사용 모델, 프롬프트 버전, 배경 해상도, 상태 등
# =========================================================

class GenRun(Base):
    """한 번의 이미지 생성 작업 단위를 나타내는 테이블(gen_runs)."""
    __tablename__ = "gen_runs"

    run_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    # jobs(job_id) FK
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("jobs.job_id"),
        nullable=True,
        index=True,
    )

    # tenants(tenant_id) 와 논리적으로 연결되는 값
    tenant_id = Column(String, nullable=False)

    # image_assets(image_asset_id) FK - 업로드 원본 이미지
    src_asset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("image_assets.image_asset_id"),
        nullable=True,
        index=True,
    )

    # image_assets(image_asset_id) FK - 누끼(cutout) 이미지
    cutout_asset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("image_assets.image_asset_id"),
        nullable=True,
        index=True,
    )

    # gen_models(model_id) FK - 사용 모델
    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("gen_models.model_id"),
        nullable=True,
        index=True,
    )

    # 프롬프트 버전
    prompt_version = Column(Text, nullable=True)

    # 배경 캔버스 크기
    bg_width = Column(Integer, nullable=True)
    bg_height = Column(Integer, nullable=True)

    # 상태: queued / running / done / failed
    status = Column(
        Text,
        nullable=True,
        server_default=text("'queued'::text"),
        index=True,
    )

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=True,
    )
    finished_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # 시퀀스 gen_runs_pk_seq 를 쓰는 auto-increment pk
    pk = Column(Integer, nullable=False)

    # run 전체 latency (ms)
    latency_ms = Column(Float, nullable=True)

    # plan.json 위치
    plan_json_url = Column(Text, nullable=True)

    # ========= relationships =========

    # 사용 모델 (gen_models)
    model = relationship("GenModel", back_populates="runs")

    # 원본 / 누끼 image_assets
    src_asset = relationship(
        "ImageAsset",
        foreign_keys=[src_asset_id],
        lazy="joined",
    )
    cutout_asset = relationship(
        "ImageAsset",
        foreign_keys=[cutout_asset_id],
        lazy="joined",
    )

    # 이 run 에서 생성된 variant 목록
    variants = relationship(
        "GenVariant",
        back_populates="run",
        order_by="GenVariant.index",
        cascade="all, delete-orphan",
    )

    # 연결된 job 정보
    job = relationship(
        "Job",
        back_populates="gen_runs",
        lazy="joined",
    )


# =========================================================
#  GEN_VARIANTS
#   - 한 run 안에서 variant1/2/3 별 결과 및 캔버스/마스크/bg 이미지 FK
# =========================================================

class GenVariant(Base):
    """각 run 에서 생성된 개별 결과 이미지(gen_variants)."""
    __tablename__ = "gen_variants"

    gen_variant_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    # gen_runs(run_id) FK
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("gen_runs.run_id"),
        nullable=True,
        index=True,
    )

    # variant 순서 (1, 2, 3 ...)
    index = Column(Integer, nullable=True)

    # image_assets(image_asset_id) FK – 캔버스 / 마스크 / 배경 결과
    canvas_asset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("image_assets.image_asset_id"),
        nullable=True,
        index=True,
    )
    mask_asset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("image_assets.image_asset_id"),
        nullable=True,
        index=True,
    )
    bg_asset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("image_assets.image_asset_id"),
        nullable=True,
        index=True,
    )

    # pbg_placement_presets(placement_preset_id) FK
    placement_preset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pbg_placement_presets.placement_preset_id"),
        nullable=True,
        index=True,
    )

    prompt_en = Column(Text, nullable=True)
    negative_en = Column(Text, nullable=True)

    # seed / steps / 추론 시간(ms)
    seed_base = Column(
        Integer,
        nullable=True,
        server_default=text("13"),
    )
    steps = Column(
        Integer,
        nullable=True,
        server_default=text("20"),
    )
    infer_ms = Column(Float, nullable=True)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=True,
    )

    # gen_variants_pk_seq 를 사용하는 auto-increment pk
    pk = Column(Integer, nullable=False)

    # 이 variant 전체 latency (ms)
    latency_ms = Column(Float, nullable=True)

    # ========= relationships =========

    # 어떤 run 에 속하는지
    run = relationship("GenRun", back_populates="variants")

    # 캔버스/마스크/결과 이미지 FK
    canvas_asset = relationship(
        "ImageAsset",
        foreign_keys=[canvas_asset_id],
        lazy="joined",
    )
    mask_asset = relationship(
        "ImageAsset",
        foreign_keys=[mask_asset_id],
        lazy="joined",
    )
    bg_asset = relationship(
        "ImageAsset",
        foreign_keys=[bg_asset_id],
        lazy="joined",
    )

    # placement preset (pbg_placement_presets)
    placement_preset = relationship(
        "PbgPlacementPreset",
        foreign_keys=[placement_preset_id],
        lazy="joined",
    )


# ======================================
#  JOBS
#   - 하나의 광고 생성 요청 단위
# ======================================

class Job(Base):
    """광고 생성 파이프라인의 상위 Job 단위(jobs)."""
    __tablename__ = "jobs"

    job_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    tenant_id = Column(String, nullable=False)
    store_id = Column(String, nullable=True)
    status = Column(Text, nullable=True, default="queued")
    version = Column(Text, nullable=True)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=True,
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

    # 현재 파이프라인 단계 (예: gen_vlm_analyze, img_gen, ...)
    current_step = Column(Text, nullable=True)

    # ========= relationships =========

    # Job 에 연결된 run 목록
    gen_runs = relationship(
        "GenRun",
        back_populates="job",
        lazy="selectin",
    )

    # Job 에서 생성된 최종 후보 이미지 목록
    variants = relationship(
        "JobVariant",
        back_populates="job",
        lazy="selectin",
        cascade="all, delete-orphan",
    )


# ======================================
#  JOB_VARIANTS
#   - Job 단위 최종 후보 이미지 목록
# ======================================

class JobVariant(Base):
    """
    jobs_variants 테이블.

    하나의 Job 에 대해 사용자에게 보여줄 개별 후보 이미지를 나타낸다.
    """
    __tablename__ = "jobs_variants"

    job_variants_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("jobs.job_id"),
        nullable=False,
    )
    img_asset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("image_assets.image_asset_id"),
        nullable=True,
    )
    creation_order = Column(Integer, nullable=True)
    selected = Column(Boolean, nullable=True, default=False)
    
    status = Column(String, nullable=True)
    current_step = Column(String, nullable=True)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=True,
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

    # ========= relationships =========

    job = relationship(
        "Job",
        back_populates="variants",
        lazy="joined",
    )
    image_asset = relationship(
        "ImageAsset",
        lazy="joined",
    )


# ======================================
#  세션 의존성
# ======================================

def get_db():
    """
    FastAPI 의존성 주입용 DB 세션 생성기.

    호출 측에서는:

        db: Session = Depends(get_db)

    형태로 사용한다.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
