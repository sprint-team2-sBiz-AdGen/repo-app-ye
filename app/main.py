#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

역할
- gen_vlm_analyze 단계: DB에서 job_id로 원본 이미지를 찾고, 누끼 생성 → GPT VLM 분석 → plan 생성
- img_gen 단계: 저장된 plan_json 기반으로 PBG 배경 이미지를 생성

파이프라인 단계 (jobs.current_step)
- gen_vlm_analyze
- img_gen

관련 DB 테이블
- image_assets
    - original  : 프론트/이전 단계에서 이미 저장됨
    - pbg_cutout: 여기서 누끼 생성 직후 저장
- job_inputs
    - job_id, img_asset_id, tone_style_id 는 프론트/이전 단계에서 생성
- gen_runs
    - /llava/analyze 시작 시점에 run 레코드 생성
      (원본/누끼 asset FK + canvas 크기 + prompt_version + job_id 포함)
"""

import io
import json
from uuid import UUID
from pathlib import Path

from fastapi import FastAPI, Depends, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.config import ASSETS_DIR, PART_NAME, ENABLE_JOB_STATE_LISTENER

from app.gen_store import (
    save_cutout_asset,
    update_job_step,
    create_gen_run,
    list_pbg_prompts_from_db,
    create_pbg_prompt_in_db,
)

from app.config import ASSETS_DIR, PART_NAME
from app.utils import save_asset, abs_from_url
from contextlib import asynccontextmanager

from app.gpt_core import (
    gpt_analyze_from_bytes,
    release_gpt_model_and_processor,
)

from app.pbg_core import (
    pbg_background_to_png_bytes,
    release_pbg_pipeline,
    release_remover,
    run_remove_background,
    finalize_plan_and_save,
)

import logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup
    print("=" * 60)
    print("애플리케이션 시작 중...")
    logger.info("애플리케이션 시작 중...")
    
    if ENABLE_JOB_STATE_LISTENER:
        print(f"ENABLE_JOB_STATE_LISTENER: {ENABLE_JOB_STATE_LISTENER}")
        try:
            from services.job_state_listener import start_listener
            print("Job State Listener 시작...")
            logger.info("Job State Listener 시작...")
            await start_listener()
            print("✓ Job State Listener 시작 완료")
        except Exception as e:
            print(f"❌ Job State Listener 시작 실패: {e}")
            logger.error(f"Job State Listener 시작 실패: {e}", exc_info=True)
    else:
        print("Job State Listener 비활성화됨")
    
    yield
    
    # Shutdown
    print("애플리케이션 종료 중...")
    logger.info("애플리케이션 종료 중...")
    
    if ENABLE_JOB_STATE_LISTENER:
        try:
            from services.job_state_listener import stop_listener
            print("Job State Listener 종료...")
            logger.info("Job State Listener 종료...")
            await stop_listener()
        except Exception as e:
            print(f"❌ Job State Listener 종료 실패: {e}")
            logger.error(f"Job State Listener 종료 실패: {e}", exc_info=True)

# =====================================================
# FastAPI & 기본 설정
# =====================================================

app = FastAPI(title=f"app-{PART_NAME}", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS_ROOT = Path(ASSETS_DIR) / "shared" / f"{PART_NAME}-app"
ASSETS_ROOT.mkdir(parents=True, exist_ok=True)


# =====================================================
# 헬스체크
# =====================================================

@app.get("/healthz")
def health(db: Session = Depends(get_db)):
    """DB 연결 여부를 확인하는 헬스체크 엔드포인트."""
    db.execute(text("SELECT 1"))
    return {"ok": True, "service": f"app-{PART_NAME}"}


@app.get("/")
def root():
    """서비스 기본 정보 확인용 루트 엔드포인트."""
    return {
        "service": f"app-{PART_NAME}",
        "message": "Feedly AdGen API - GPT VLM + PBG",
        "status": "ok",
    }


# =====================================================
# GPT VLM 분석 + plan 생성 엔드포인트 (gen_vlm_analyze 단계)
#
# 입력:
#   - job_id
#   - canvas_width / canvas_height
#   - pbg_prompt_version
#
# 처리 순서:
#   1) job_inputs → image_assets → jobs 조인으로
#      - 원본 이미지 URL
#      - tenant_id
#      - tone_style_id(= strategy_name) 조회
#   2) Remover 로 누끼 딴 RGBA 생성, image_type="pbg_cutout" 로 image_assets 저장
#   3) gen_runs 레코드 생성 (job_id FK 포함)
#   4) 누끼 이미지를 GPT VLM 에 넘겨 skeleton plan 생성
#   5) finalize_plan_and_save 로 plan.json 저장 + 메타데이터 채우기
#   6) jobs.current_step='gen_vlm_analyze', status='done' 으로 업데이트
#   7) 최종 plan 을 그대로 JSON 으로 응답
# =====================================================

@app.post("/llava/analyze")
async def gpt_analyze_endpoint(
    job_id: str = Form(...),
    canvas_width: int = Form(1080),
    canvas_height: int = Form(1080),
    pbg_prompt_version: str = Form("v1"),
    db: Session = Depends(get_db),
):
    """
    gen_vlm_analyze 단계.

    - job_id 기준으로 원본 이미지와 스타일 정보를 조회하고
    - 누끼 + GPT VLM 분석 결과를 기반으로 plan 을 생성한 뒤
    - plan.json 을 저장하고 jobs/gen_runs 상태를 갱신한다.
    """
    # -------------------------------------------------
    # 0) job_id 문자열 → UUID 변환
    # -------------------------------------------------
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 job_id 형식입니다 (uuid 아님).")

    # -------------------------------------------------
    # 1) job_id 로 원본 이미지 / tenant / tone_style_id 정보 조회
    #    job_inputs → image_assets → jobs
    # -------------------------------------------------
    row = (
        db.execute(
            text(
                """
                SELECT
                    ji.img_asset_id,
                    ia.image_url,
                    j.tenant_id,
                    ji.tone_style_id,
                    ia.creator_id
                FROM job_inputs ji
                JOIN image_assets ia
                  ON ji.img_asset_id = ia.image_asset_id
                JOIN jobs j
                  ON ji.job_id = j.job_id
                WHERE ji.job_id = :job_id
                LIMIT 1
                """
            ),
            {"job_id": job_uuid},
        )
        .mappings()
        .first()
    )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"job_id={job_id} 에 대한 job_inputs/image_assets/jobs 정보를 찾을 수 없습니다.",
        )

    src_asset_id = row["img_asset_id"]
    uploaded_url = row["image_url"]
    tenant_id = row["tenant_id"]
    strategy_name = row["tone_style_id"]
    creator_id    = row["creator_id"]

    if not tenant_id:
        raise HTTPException(
            status_code=500,
            detail=f"job_id={job_id} 에 대한 tenant_id 가 jobs 테이블에 없습니다.",
        )

    if not strategy_name:
        raise HTTPException(
            status_code=400,
            detail=f"job_id={job_id} 에 대한 tone_style_id 가 job_inputs 에 없습니다.",
        )

    # URL → 실제 파일 경로 변환
    img_path = abs_from_url(uploaded_url)

    try:
        orig_img = Image.open(img_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"원본 이미지 로드 실패: {e}",
        )

    # PIL 모드 정리
    if orig_img.mode not in ("RGB", "RGBA"):
        orig_img = orig_img.convert("RGB")

    filename = f"{src_asset_id}.png"

    # -------------------------------------------------
    # 2) gen_vlm_analyze 단계 시작: job 상태 running 으로 전환
    # -------------------------------------------------
    try:
        update_job_step(
            db,
            job_id=job_uuid,
            current_step="gen_vlm_analyze",
            status="running",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update job to gen_vlm_analyze/running: {e}",
        )

    # -------------------------------------------------
    # 3) 누끼 생성 + pbg_cutout 저장
    # -------------------------------------------------
    orig_for_remove = orig_img.convert("RGB")
    cut_rgba = run_remove_background(orig_for_remove)

    cut_info = save_cutout_asset(
        db,
        tenant_id=tenant_id,
        pil_image=cut_rgba,
        job_id=job_uuid,
        creator_id=creator_id,
    )
    cutout_asset_id = cut_info["asset_id"]
    cutout_url = cut_info["url"]

    # -------------------------------------------------
    # 4) gen_runs 에 run 생성 (job_uuid 와 연결)
    # -------------------------------------------------
    try:
        run_id = create_gen_run(
            db=db,
            tenant_id=tenant_id,
            src_asset_id=src_asset_id,
            cutout_asset_id=cutout_asset_id,
            prompt_version=pbg_prompt_version,
            bg_width=canvas_width,
            bg_height=canvas_height,
            model_id=None,
            job_id=job_uuid,
        )
    except Exception as e:
        update_job_step(
            db,
            job_id=job_uuid,
            status="failed",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create gen_run: {e}",
        )

    # -------------------------------------------------
    # 5) GPT VLM 분석 – 누끼 이미지를 PNG bytes 로 변환해서 전달
    # -------------------------------------------------
    buf = io.BytesIO()
    cut_rgba.convert("RGB").save(buf, format="PNG")
    cut_bytes = buf.getvalue()

    try:
        plan = gpt_analyze_from_bytes(
            image_bytes=cut_bytes,
            filename=filename,
            strategy_name=str(strategy_name),
            width=canvas_width,
            height=canvas_height,
            pbg_prompt_version=pbg_prompt_version,
            db=db,
            gpt_prompt_uid=None,
        )
    except Exception as e:
        update_job_step(
            db,
            job_id=job_uuid,
            status="failed",
        )
        raise HTTPException(
            status_code=500,
            detail=f"GPT VLM analyze failed: {e}",
        )

    # plan 안에 job/run/tenant 메타 붙이기
    plan["gen_run_id"] = str(run_id)
    plan["job_id"] = str(job_uuid)
    plan["tenant_id"] = tenant_id

    # -------------------------------------------------
    # 6) plan 최종 정리 + plan.json 저장
    # -------------------------------------------------
    final_plan = finalize_plan_and_save(
        plan=plan,
        db=db,
        tenant_id=tenant_id,
        src_asset_id=src_asset_id,
        cutout_asset_id=cutout_asset_id,
        uploaded_url=uploaded_url,
        cutout_url=cutout_url,
        run_id=run_id,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        pbg_prompt_version=creator_id,
        creator_id=creator_id,
    )

    final_plan["job_id"] = str(job_uuid)
    final_plan["gen_run_id"] = str(run_id)
    final_plan["tenant_id"] = tenant_id

    # -------------------------------------------------
    # 7) gen_vlm_analyze 단계 완료 (done)
    #    → Listener는 여기 상태를 보고 다음 단계(img_gen) 트리거
    # -------------------------------------------------
    update_job_step(
        db,
        job_id=job_uuid,
        current_step="gen_vlm_analyze",
        status="done",
    )

    # 8) 최종 plan 응답
    return JSONResponse(final_plan)


# =====================================================
# PBG 배경 생성 엔드포인트 (img_gen 단계)
#
# 입력:
#   - job_id
#
# 처리 순서:
#   1) gen_runs 에서 해당 job_id 의 최신 run + plan_json_url 조회
#   2) plan.json 로드 → plan_obj 구성
#   3) jobs 에서 tenant_id 조회
#   4) jobs.current_step='img_gen', status='running' 으로 변경
#   5) variant1, variant2, variant3 순서로 PBG 실행
#      - 각 variant 는 pbg_background_to_png_bytes 내부에서
#        image_assets / gen_variants / job_variants insert
#      - style_key == "variant3" 일 때
#        finish_gen_run + jobs.current_step='img_gen', status='done' 처리
#   6) 프론트는 이미지를 직접 사용하지 않으므로, 바디 없이 204 반환
# =====================================================

@app.post("/generate")
async def pbg_generate_endpoint(
    job_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    DB 리스너/백엔드에서 호출하는 img_gen 단계용 엔드포인트.

    - plan.json 기반으로 variant1~3 이미지를 생성하고
    - gen_runs / jobs 상태를 갱신한 뒤
    - HTTP 204(No Content) 로 응답한다.
    """
    # ---------------------------------------------
    # 1) job_id 검증
    # ---------------------------------------------
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="잘못된 job_id 형식입니다 (uuid 아님).",
        )

    # ---------------------------------------------
    # 2) gen_runs 에서 최신 run + plan_json_url 조회
    # ---------------------------------------------
    run_row = (
        db.execute(
            text(
                """
                SELECT run_id, plan_json_url
                FROM gen_runs
                WHERE job_id = :job_id
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            {"job_id": job_uuid},
        )
        .mappings()
        .first()
    )

    if run_row is None or not run_row["plan_json_url"]:
        raise HTTPException(
            status_code=404,
            detail="해당 job에 대한 plan_json을 찾을 수 없습니다.",
        )

    plan_path = abs_from_url(run_row["plan_json_url"])
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_obj = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"plan.json 로드 실패: {e}",
        )

    # job / run 정보 보정
    plan_obj.setdefault("job_id", str(job_uuid))
    plan_obj.setdefault("gen_run_id", str(run_row["run_id"]))

    # 문자열로 만들어서 PBG 에 넘김
    plan_payload = json.dumps(plan_obj, ensure_ascii=False)

    # ---------------------------------------------
    # 3) jobs 에서 tenant_id 조회
    # ---------------------------------------------
    tenant_row = db.execute(
        text(
            """
            SELECT tenant_id
            FROM jobs
            WHERE job_id = :job_id
            """
        ),
        {"job_id": job_uuid},
    ).fetchone()

    if tenant_row is None or not tenant_row[0]:
        raise HTTPException(
            status_code=404,
            detail="해당 job에 대한 tenant_id를 jobs 테이블에서 찾을 수 없습니다.",
        )

    tenant_id = tenant_row[0]

    # img_gen 시작 → running
    update_job_step(
        db,
        job_id=job_uuid,
        current_step="img_gen",
        status="running",
    )

    # ---------------------------------------------
    # 4) PBG 실행 – variant1~3 모두 생성
    #    seed 는 plan.background_prompts 에 저장된 값을 사용한다.
    # ---------------------------------------------
    try:
        for vk in ("variant1", "variant2", "variant3"):
            # 이 함수 내부에서:
            # - canvas/mask 저장
            # - gen_variants / job_variants insert
            # - style_key == "variant3" 인 경우
            #     finish_gen_run + jobs.current_step='img_gen', status='done' 처리
            pbg_background_to_png_bytes(
                plan_json=plan_payload,
                style_key=vk,
                db=db,
                seed=None,       # seed 는 plan 쪽 seed 사용
                tenant_id=tenant_id,
            )

        # 프론트에서 이미지를 직접 쓰지 않으므로, 바디 없이 204 반환
        return Response(status_code=204)

    except Exception as e:
        # 실패 시 job 상태 failed
        update_job_step(
            db,
            job_id=job_uuid,
            status="failed",
        )
        raise HTTPException(status_code=500, detail=f"PBG generate failed: {e}")


# =====================================================
# 모델 언로드 엔드포인트
# =====================================================

@app.post("/models/release")
def release_models():
    """
    PBG / Remover / GPT VLM 모델을 수동으로 언로드하는 엔드포인트.
    장시간 구동 후 메모리를 비워줄 때 사용한다.
    """
    release_pbg_pipeline()
    release_remover()
    release_gpt_model_and_processor()
    return {"ok": True}


# =====================================================
# 스타일별 프롬프트/버전 조회 API
# =====================================================

@app.get("/pbg/prompts")
def list_pbg_prompts(
    prompt_type: str,
    db: Session = Depends(get_db),
):
    """
    특정 prompt_type 에 대해 다음 정보를 반환한다.

    - 사용 가능한 모든 prompt_version 목록
    - 각 버전의 prompt / negative_prompt (JSONB)
    """
    try:
        return list_pbg_prompts_from_db(db, prompt_type)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load prompts for '{prompt_type}': {e}",
        )


# =====================================================
# 요청 바디용 Pydantic 모델
# =====================================================

class PBGPromptCreateRequest(BaseModel):
    prompt_type: str       # 예: "Hero Dish Focus"
    prompt_version: str    # 예: "v1", "v2"
    prompt_en: str
    negative_en: str


# =====================================================
# 프롬프트 추가
# =====================================================

@app.post("/pbg/prompts")
def create_pbg_prompt(
    req: PBGPromptCreateRequest,
    db: Session = Depends(get_db),
):
    """
    새로운 PBG 프롬프트 버전을 DB 에 추가하는 엔드포인트.
    이미 동일한 (prompt_type, prompt_version)이 존재하면 400 을 반환한다.
    """
    try:
        new_asset = create_pbg_prompt_in_db(
            db,
            prompt_type=req.prompt_type,
            prompt_version=req.prompt_version,
            prompt_en=req.prompt_en,
            negative_en=req.negative_en,
        )
    except ValueError as e:
        # gen_store 쪽에서 "이미 존재" 같은 비즈니스 에러를 ValueError 로 던진다.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create prompt: {e}",
        )

    return {
        "prompt_asset_id": str(new_asset.prompt_asset_id),
        "prompt_type": new_asset.prompt_type,
        "prompt_version": new_asset.prompt_version,
    }
