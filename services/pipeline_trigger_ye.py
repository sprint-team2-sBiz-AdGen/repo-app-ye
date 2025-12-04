#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YE Pipeline Trigger

역할
- jobs.current_step / status 를 보고 YE 파이프라인 다음 단계를 자동으로 트리거

YE 파이프라인 규칙
- (current_step='user_img_input',   status='done') → /llava/analyze  (gen_vlm_analyze 단계 시작)
- (current_step='gen_vlm_analyze', status='done') → /generate       (img_gen 단계 시작)

특징
- JobStateListenerYE 에서 이 모듈만 호출
- HTTP 호출은 여기서만 처리
"""

import logging
from typing import Optional

import asyncpg
import httpx
import uuid

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

# 이 FastAPI 앱이 떠 있는 주소 (uvicorn 실행 기준)
API_HOST = "127.0.0.1"
API_PORT = 8010
BASE_URL = f"http://{API_HOST}:{API_PORT}"

# YE 파트용 단계 매핑
# (current_step, status) → 다음에 호출할 API 정보
PIPELINE_STAGES_YE = {
    ("user_img_input", "done"): {
        "name": "gen_vlm_analyze",
        "api_endpoint": "/llava/analyze",
        "kind": "llava_analyze",
    },
    ("gen_vlm_analyze", "done"): {
        "name": "img_gen",
        "api_endpoint": "/generate",
        "kind": "pbg_generate",
    },
}


async def trigger_next_pipeline_stage_ye(
    job_id: str,
    current_step: Optional[str],
    status: str,
    tenant_id: Optional[str],
) -> None:
    """
    YE 파이프라인 다음 단계 트리거

    - current_step / status 기준으로 PIPELINE_STAGES_YE 에서 다음 단계 조회
    - Job 상태를 한 번 더 확인 (_verify_job_state_ye) 해서 중복 실행 방지
    - 해당 API 엔드포인트에 HTTP 호출
    """
    # 트리거 조건 확인
    if not current_step or status != "done":
        logger.debug(
            f"[YE TRIGGER] 조건 불만족 → 스킵: job_id={job_id}, "
            f"current_step={current_step}, status={status}"
        )
        return

    stage_info = PIPELINE_STAGES_YE.get((current_step, status))
    if not stage_info:
        logger.debug(
            f"[YE TRIGGER] YE 파이프라인 대상 아님 → 스킵: job_id={job_id}, "
            f"current_step={current_step}, status={status}"
        )
        return

    next_name = stage_info["name"]
    api_endpoint = stage_info["api_endpoint"]
    kind = stage_info["kind"]

    # 중복 실행 방지: 실제 DB 상태 재확인
    if not await _verify_job_state_ye(job_id, current_step, status):
        logger.info(
            f"[YE TRIGGER] Job 상태가 이미 달라져서 스킵: job_id={job_id}, "
            f"expected_step={current_step}, expected_status={status}"
        )
        return

    url = f"{BASE_URL}{api_endpoint}"

    # 단계별로 전송할 Form 데이터 구성
    if kind == "llava_analyze":
        data = {
            "job_id": job_id,
            "canvas_width": 1080,
            "canvas_height": 1080,
            "pbg_prompt_version": "v1",
        }
    elif kind == "pbg_generate":
        data = {"job_id": job_id}
    else:
        logger.error(
            f"[YE TRIGGER] 알 수 없는 kind='{kind}' → 스킵: job_id={job_id}"
        )
        return

    print(f"[YE TRIGGER] 파이프라인 단계 트리거: job_id={job_id}, next={next_name}")
    logger.info(
        f"[YE TRIGGER] 파이프라인 단계 트리거: job_id={job_id}, "
        f"current_step={current_step}, status={status}, "
        f"next={next_name}, api={url}"
    )

    try:
        # gpt_analyze_endpoint / pbg_generate_endpoint 모두 Form(...) 이라
        # json 이 아니라 data 로 보내야 함
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, data=data)
            # /generate 는 204, /llava/analyze 는 200(또는 201) 예상
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{api_endpoint} 실패: {resp.status_code} {resp.text}"
                )

        logger.info(
            f"[YE TRIGGER] 파이프라인 단계 실행 성공: job_id={job_id}, next={next_name}"
        )
    except Exception as e:
        logger.error(
            f"[YE TRIGGER] 파이프라인 단계 실행 실패: job_id={job_id}, "
            f"next={next_name}, error={e}",
            exc_info=True,
        )
        # 여기서는 예외를 위로 올리지 않음 (리스너는 계속 돌아가야 함)


async def _verify_job_state_ye(
    job_id: str,
    expected_step: str,
    expected_status: str,
) -> bool:
    """
    YE 용 Job 상태 재확인 (중복 실행 방지)

    - jobs.current_step / jobs.status 가 여전히 우리가 기대하는 값인지 확인
    """
    asyncpg_url = DATABASE_URL.replace("postgresql://", "postgres://")

    try:
        conn = await asyncpg.connect(asyncpg_url)
    except Exception as e:
        logger.error(f"[YE VERIFY] DB 연결 실패: {e}", exc_info=True)
        # 검증 실패 시, 일단은 실행하지 않는 쪽으로
        return False

    try:
        row = await conn.fetchrow(
            """
            SELECT current_step, status
            FROM jobs
            WHERE job_id = $1
            """,
            uuid.UUID(job_id),
        )

        if not row:
            logger.warning(f"[YE VERIFY] Job을 찾을 수 없음: job_id={job_id}")
            return False

        cur_step = row["current_step"]
        cur_status = row["status"]

        if cur_step == expected_step and cur_status == expected_status:
            return True

        logger.debug(
            f"[YE VERIFY] Job 상태 불일치: job_id={job_id}, "
            f"expected_step={expected_step}, expected_status={expected_status}, "
            f"actual_step={cur_step}, actual_status={cur_status}"
        )
        return False

    except Exception as e:
        logger.error(f"[YE VERIFY] Job 상태 확인 오류: {e}", exc_info=True)
        return False
    finally:
        try:
            await conn.close()
        except Exception:
            pass
