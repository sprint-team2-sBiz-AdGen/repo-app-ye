#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_store.py
- image_assets / gen_runs / gen_variants / jobs / job_variants / pbg_prompt_assets 관련 CRUD 헬퍼

저장 규칙:
image_assets.image_type:
  - "original"     : 업로드 원본 이미지
  - "pbg_cutout"   : remover로 누끼 딴 이미지
  - "pbg_canvas"   : 피사체를 배치한 캔버스(배경 inpaint 입력용)
  - "pbg_mask"     : inpaint용 마스크 (255=배경, 0=피사체)
  - "pbg_generated": PBG 결과 이미지
"""

import os
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.utils import save_asset
from app.database import (
    ImageAsset,
    GenRun,
    GenVariant,
    GenModel,
    PbgPromptAsset,
    Job,
    JobVariant,
)

# =========================================================
# 공통 헬퍼
# =========================================================

def _ensure_uuid(v) -> uuid.UUID:
    """str / uuid.UUID 어떤 게 와도 uuid.UUID 로 통일"""
    if isinstance(v, uuid.UUID):
        return v
    return uuid.UUID(str(v))


def _next_seq_val(db: Session, seq_name: str) -> int:
    """
    주어진 시퀀스에서 다음 값 가져오기
    - seq_name 은 내부에서 정해둔 시퀀스 이름만 넣을 것
    """
    val = db.execute(
        text(f"SELECT nextval('{seq_name}')")
    ).scalar_one()
    return int(val)


# =========================================================
# 업로드 원본 / 누끼 저장
# =========================================================

def _insert_image_asset(
    db: Session,
    *,
    tenant_id: str,
    image_type: str,
    asset_meta: Dict[str, Any],
    job_id: uuid.UUID | None = None,
    creator_id: uuid.UUID | str | None = None,
) -> uuid.UUID:
    """
    save_asset() 결과(meta)를 받아 insert
    """
    asset_id = _ensure_uuid(asset_meta["asset_id"])
    pk = _next_seq_val(db, "image_assets_pk_seq")

    row = ImageAsset(
        image_asset_id=asset_id,
        image_type=image_type,
        image_url=asset_meta["url"],
        mask_url=None,
        width=int(asset_meta["width"]),
        height=int(asset_meta["height"]),
        creator_id=_ensure_uuid(creator_id) if creator_id else None,
        tenant_id=tenant_id,
        job_id=job_id,
        pk=pk,
    )

    db.add(row)
    db.commit()
    print(f"[gen_store] ✅ image_assets insert: type={image_type}, url={asset_meta['url']}")
    return asset_id


def save_original_asset(
    db: Session,
    tenant_id: str,
    pil_image: Image.Image,
    job_id: uuid.UUID | str | None = None,
) -> Dict[str, Any]:
    """
    업로드 원본 저장
    """
    meta = save_asset(
        tenant_id=tenant_id,
        kind="uploaded",
        image=pil_image,
        ext=".png",
    )
    asset_id = _insert_image_asset(
        db,
        tenant_id=tenant_id,
        image_type="original",
        asset_meta=meta,
        job_id=job_id,
    )
    meta["asset_id"] = str(asset_id)
    return meta


def save_cutout_asset(
    db: Session,
    tenant_id: str,
    pil_image: Image.Image,
    job_id: uuid.UUID | str | None = None,
    creator_id: uuid.UUID | str | None = None,
) -> Dict[str, Any]:
    """
    remover 로 누끼 딴 이미지 저장
    """
    meta = save_asset(
        tenant_id=tenant_id,
        kind="cutout",
        image=pil_image,
        ext=".png",
    )
    asset_id = _insert_image_asset(
        db,
        tenant_id=tenant_id,
        image_type="pbg_cutout",
        asset_meta=meta,
        job_id=job_id,
        creator_id=creator_id,
    )
    meta["asset_id"] = str(asset_id)
    return meta


# =========================================================
# 캔버스 / 마스크 / 결과 이미지 저장
# =========================================================

def save_canvas_and_mask_assets(
    db: Session,
    tenant_id: str,
    canvas_img: Image.Image,
    mask_img: Image.Image,
    job_id: uuid.UUID | str | None = None,
    creator_id: uuid.UUID | str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    PBG 입력용 캔버스 + 마스크 저장
    """
    canvas_meta = save_asset(
        tenant_id=tenant_id,
        kind="canvas",
        image=canvas_img,
        ext=".png",
    )
    mask_meta = save_asset(
        tenant_id=tenant_id,
        kind="mask",
        image=mask_img,
        ext=".png",
    )

    try:
        canvas_id = _insert_image_asset(
            db,
            tenant_id=tenant_id,
            image_type="pbg_canvas",
            asset_meta=canvas_meta,
            job_id=job_id,
            creator_id=creator_id,
        )
        mask_id = _insert_image_asset(
            db,
            tenant_id=tenant_id,
            image_type="pbg_mask",
            asset_meta=mask_meta,
            job_id=job_id,
            creator_id=creator_id,
        )
    except Exception as e:
        raise RuntimeError(f"save_canvas_and_mask_assets DB insert 실패: {e}")

    canvas_meta["asset_id"] = str(canvas_id)
    mask_meta["asset_id"] = str(mask_id)
    return canvas_meta, mask_meta


def save_generated_asset(
    db: Session,
    tenant_id: str,
    generated_img: Image.Image,
    creator_id: uuid.UUID | str | None = None,
    job_id: uuid.UUID | str | None = None,
) -> Dict[str, Any]:
    """
    PBG 결과 이미지 저장
    """
    gen_meta = save_asset(
        tenant_id=tenant_id,
        kind="generated",
        image=generated_img,
        ext=".png",
    )
    try:
        gen_id = _insert_image_asset(
            db,
            tenant_id=tenant_id,
            image_type="pbg_generated",
            asset_meta=gen_meta,
            creator_id=creator_id,
            job_id=job_id,
        )
    except Exception as e:
        raise RuntimeError(f"save_generated_asset DB insert 실패: {e}")

    gen_meta["asset_id"] = str(gen_id)
    return gen_meta


# =========================================================
# GEN_MODELS (PBG 기본 모델 선택)
# =========================================================

def _get_default_pbg_model_id(db: Session) -> uuid.UUID | None:
    """
    기본 PBG 모델 하나 골라서 model_id 리턴.
    """
    env_id = os.getenv("PBG_MODEL_ID")
    if env_id:
        try:
            return uuid.UUID(env_id)
        except Exception:
            print(f"[gen_store] ⚠ invalid PBG_MODEL_ID={env_id}, ignore")

    repo = os.getenv("PBG_MODEL_REPO", "yahoo-inc/photo-background-generation")
    row = (
        db.query(GenModel)
        .filter(GenModel.repo == repo)
        .order_by(GenModel.created_at.desc())
        .first()
    )

    if row is None:
        print(f"[gen_store] ⚠ default PBG model not found for repo='{repo}'")
        return None

    return row.model_id


# =========================================================
# GEN_RUNS / GEN_VARIANTS
# =========================================================

def create_gen_run(
    db: Session,
    tenant_id: str,
    src_asset_id: uuid.UUID,
    cutout_asset_id: uuid.UUID,
    prompt_version: str,
    bg_width: int,
    bg_height: int,
    model_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,  # job FK
) -> uuid.UUID:
    """
    gen_runs 한 건 insert
    """
    if model_id is None:
        model_id = _get_default_pbg_model_id(db)

    pk_val = _next_seq_val(db, "gen_runs_pk_seq")

    run = GenRun(
        tenant_id=tenant_id,
        src_asset_id=src_asset_id,
        cutout_asset_id=cutout_asset_id,
        prompt_version=prompt_version,
        bg_width=bg_width,
        bg_height=bg_height,
        model_id=model_id,
        status="queued",
        pk=pk_val,
        job_id=job_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run.run_id


def update_gen_run_status(
    db: Session,
    run_id: uuid.UUID | str,
    status: str,
    latency_ms: float | None = None,
) -> None:
    """
    gen_runs.status / latency_ms 갱신.
    - gen_runs 테이블에는 updated_at 컬럼이 없으므로 건드리지 않음.
    - SQL 에러가 나면 rollback 하고 에러만 로그로 남김.
    """
    rid = _ensure_uuid(run_id)

    try:
        if latency_ms is not None:
            db.execute(
                text(
                    """
                    UPDATE gen_runs
                    SET status = :status,
                        latency_ms = :latency_ms
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "status": status,
                    "latency_ms": float(latency_ms),
                    "run_id": str(rid),
                },
            )
        else:
            db.execute(
                text(
                    """
                    UPDATE gen_runs
                    SET status = :status
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "status": status,
                    "run_id": str(rid),
                },
            )

        db.commit()
    except SQLAlchemyError as e:
        # 트랜잭션 깨진 상태 정리
        db.rollback()
        print(f"[gen_store] ⚠ update_gen_run_status 실패: {e}")
        return

    print(
        f"[gen_store] ✅ gen_runs status update: "
        f"run_id={run_id}, status={status}, latency_ms={latency_ms}"
    )

def finish_gen_run(
    db: Session,
    *,
    run_id: str,
    status: str = "succeeded",
    latency_ms: float | None = None,
) -> None:
    """
    GEN_RUNS 상태 업데이트 + finished_at 찍기.
    """
    update_gen_run_status(db, run_id=run_id, status=status, latency_ms=latency_ms)


def create_gen_variant(
    db: Session,
    run_id: uuid.UUID | str,
    index: int,
    canvas_asset_id: uuid.UUID | str,
    mask_asset_id: uuid.UUID | str,
    bg_asset_id: uuid.UUID | str,
    placement_preset_id: Optional[uuid.UUID | str],
    prompt_en: Optional[str],
    negative_en: Optional[str],
    seed_base: int,
    steps: int,
    infer_ms: float,
    latency_ms: float | None = None,
) -> GenVariant:
    """
    gen_variants 테이블에 variant insert
    """
    pk_val = _next_seq_val(db, "gen_variants_pk_seq")

    variant = GenVariant(
        run_id=_ensure_uuid(run_id),
        index=index,
        canvas_asset_id=_ensure_uuid(canvas_asset_id),
        mask_asset_id=_ensure_uuid(mask_asset_id),
        bg_asset_id=_ensure_uuid(bg_asset_id),
        placement_preset_id=_ensure_uuid(placement_preset_id) if placement_preset_id else None,
        prompt_en=prompt_en,
        negative_en=negative_en,
        seed_base=seed_base,
        steps=steps,
        infer_ms=infer_ms,
        pk=pk_val,
        latency_ms=latency_ms,
    )

    db.add(variant)
    db.commit()
    db.refresh(variant)

    print(
        f"[gen_store] ✅ gen_variants insert: "
        f"run_id={run_id}, index={index}, placement_preset_id={placement_preset_id}, "
        f"latency_ms={latency_ms}"
    )
    return variant


# =========================================================
# JOBS / JOB_VARIANTS
# =========================================================

def create_job(
    db: Session,
    *,
    tenant_id: str,
    store_id: Optional[uuid.UUID] = None,
    status: str = "queued",
    current_step: Optional[str] = "gpt_analyze",
) -> uuid.UUID:
    """
    jobs 에 한 건 insert.
    """
    job = Job(
        tenant_id=tenant_id,
        store_id=store_id,
        status=status,
        current_step=current_step,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    print(f"[gen_store] ✅ jobs insert: job_id={job.job_id}, tenant_id={tenant_id}")
    return job.job_id


def update_job_step(
    db: Session,
    *,
    job_id: uuid.UUID | str,
    current_step: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    """
    jobs.current_step / status / updated_at 업데이트 헬퍼.
    updated_at = CURRENT_TIMESTAMP (DB 타임존 기준)

    - 트랜잭션이 aborted 상태일 때 한 번 rollback 후 재시도.
    """
    jid = _ensure_uuid(job_id)

    # 바꿀 필드 없으면 그냥 리턴
    set_clauses: List[str] = []
    params: dict[str, Any] = {"job_id": str(jid)}

    if current_step is not None:
        set_clauses.append("current_step = :current_step")
        params["current_step"] = current_step

    if status is not None:
        set_clauses.append("status = :status")
        params["status"] = status

    if not set_clauses:
        print(f"[gen_store] ⚠ update_job_step: 변경할 필드 없음 (job_id={job_id})")
        return

    # updated_at 항상 CURRENT_TIMESTAMP 로
    set_clauses.append("updated_at = CURRENT_TIMESTAMP")

    sql = f"""
        UPDATE jobs
        SET {", ".join(set_clauses)}
        WHERE job_id = :job_id
    """

    def _do_update() -> None:
        db.execute(text(sql), params)
        db.commit()

    try:
        _do_update()
    except SQLAlchemyError as e:
        msg = str(getattr(e, "orig", e))
        print(f"[gen_store] ⚠ update_job_step 1차 실패: {msg}")
        db.rollback()

        # 트랜잭션 aborted 상태면 한 번만 재시도
        if "current transaction is aborted" in msg:
            try:
                _do_update()
                print(
                    f"[gen_store] ✅ update_job_step 재시도 성공: "
                    f"job_id={job_id}, current_step={current_step}, status={status}"
                )
                return
            except SQLAlchemyError as e2:
                print(f"[gen_store] ❌ update_job_step 재시도 실패: {e2}")
                raise
        else:
            # 다른 SQL 에러면 그대로 올려보냄
            raise

    print(
        f"[gen_store] ✅ jobs update: job_id={job_id}, "
        f"current_step={current_step}, status={status}"
    )


def create_job_variant(
    db: Session,
    *,
    job_id: uuid.UUID | str,
    img_asset_id: uuid.UUID | str,
    creation_order: int,
    selected: Optional[bool] = None,
) -> JobVariant:
    """
    job_variants 테이블에 variant 1개 insert.
    - status='queued'
    - current_step='vlm_analyze'
    - insert 직후 1초 대기 후 updated_at = CURRENT_TIMESTAMP 업데이트 (트리거 발동)
    """
    variant = JobVariant(
        job_id=_ensure_uuid(job_id),
        img_asset_id=_ensure_uuid(img_asset_id),
        creation_order=creation_order,
        selected=selected,
        status="queued",
        current_step="vlm_analyze",
    )

    # 1️⃣ INSERT
    db.add(variant)
    db.commit()
    db.refresh(variant)

    # 2️⃣ 1초 대기 (심리적 안정용 + 트리거 커밋 간격 확보)
    # time.sleep(1.0)

    # 3️⃣ UPDATE → 트리거 발동
    db.execute(
        text("""
            UPDATE jobs_variants
            SET updated_at = CURRENT_TIMESTAMP
            WHERE job_variants_id = :job_variants_id
        """),
        {"job_variants_id": str(variant.job_variants_id)},
    )
    db.commit()
    db.refresh(variant)

    print(
        f"[gen_store] ✅ job_variants insert: "
        f"job_id={job_id}, img_asset_id={img_asset_id}, "
        f"creation_order={creation_order}, status=queued, current_step=vlm_analyze"
    )
    return variant

# =========================================================
# GEN_PROMPT_ASSET
# =========================================================

def list_pbg_prompts_from_db(db: Session, prompt_type: str) -> Dict[str, Any]:
    """
    DB에서 특정 스타일(prompt_type) 프롬프트 리스트 조회
    """
    rows: List[PbgPromptAsset] = (
        db.query(PbgPromptAsset)
        .filter(PbgPromptAsset.prompt_type == prompt_type)
        .order_by(PbgPromptAsset.prompt_version.asc(), PbgPromptAsset.pk.asc())
        .all()
    )

    items: List[Dict[str, Any]] = []
    for row in rows:
        prompt_dict = row.prompt or {}
        neg_dict = row.negative_prompt or {}

        prompt_en = prompt_dict.get("en") if isinstance(prompt_dict, dict) else None
        negative_en = neg_dict.get("en") if isinstance(neg_dict, dict) else None

        items.append(
            {
                "prompt_asset_id": str(row.prompt_asset_id),
                "prompt_type": row.prompt_type,
                "prompt_version": row.prompt_version,
                "prompt_en": prompt_en,
                "negative_en": negative_en,
                "prompt": prompt_dict,
                "negative_prompt": neg_dict,
            }
        )

    versions = sorted({item["prompt_version"] for item in items if item["prompt_version"]})

    return {
        "prompt_type": prompt_type,
        "versions": versions,
        "items": items,
    }


# =========================================================
# PBG_PROMPT_ASSETS insert
# =========================================================

def create_pbg_prompt_in_db(
    db: Session,
    *,
    prompt_type: str,
    prompt_version: str,
    prompt_en: str,
    negative_en: str,
) -> PbgPromptAsset:
    """
    DB에 PBG 프롬프트 insert
    """
    exists = (
        db.query(PbgPromptAsset)
        .filter(
            PbgPromptAsset.prompt_type == prompt_type,
            PbgPromptAsset.prompt_version == prompt_version,
        )
        .first()
    )
    if exists:
        raise ValueError(
            f"prompt_type={prompt_type}, prompt_version={prompt_version} 이미 존재"
        )

    pk_val = _next_seq_val(db, "pbg_prompt_assets_pk_seq")

    new_asset = PbgPromptAsset(
        prompt_asset_id=uuid.uuid4(),
        tone_style_id=None,
        prompt_type=prompt_type,
        prompt_version=prompt_version,
        prompt={"en": prompt_en},
        negative_prompt={"en": negative_en},
        pk=pk_val,
    )

    db.add(new_asset)
    db.commit()
    db.refresh(new_asset)
    return new_asset
