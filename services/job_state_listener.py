#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YE Job State Listener

역할
- current_step = 'user_img_input'  AND status = 'done' → /llava/analyze 호출 (gen_vlm_analyze 단계 시작)
- current_step = 'gen_vlm_analyze' AND status = 'done' → /generate 호출 (img_gen 단계 시작)

구조
- PostgreSQL LISTEN/NOTIFY (job_state_changed 채널)
- 서버 시작 시 1회 초기 스캔 (_initial_scan)
- 실질적인 HTTP 호출은 services.pipeline_trigger_ye.trigger_next_pipeline_stage_ye 에서 처리
"""

import asyncio
import json
import logging
from typing import Optional, Set

import asyncpg

from app.config import DATABASE_URL, JOB_STATE_LISTENER_RECONNECT_DELAY
from services.pipeline_trigger_ye import trigger_next_pipeline_stage_ye

logger = logging.getLogger(__name__)


class JobStateListenerYE:
    """YE 파트용 Job 상태 리스너"""

    def __init__(self) -> None:
        self.conn: Optional[asyncpg.Connection] = None
        self.running: bool = False
        self.reconnect_delay: int = JOB_STATE_LISTENER_RECONNECT_DELAY

        # 실행 중인 비동기 작업
        self.pending_tasks: Set[asyncio.Task] = set()
        # 동시에 같은 이벤트를 두 번 치는 것 방지 (처리 중)
        #  - key: f"{job_id}:{current_step}:{status}"
        self.active_events: Set[str] = set()
        # 이미 한 번 처리 완료된 이벤트 (재진입 방지)
        self.processed_events: Set[str] = set()

    # --------------------------------------------------
    # 외부에서 호출하는 시작 / 종료
    # --------------------------------------------------
    async def start(self) -> None:
        """리스너 메인 엔트리 (초기 스캔 + LISTEN 루프 실행)"""
        if self.running:
            return

        self.running = True
        logger.info("YE Job State Listener 시작")

        # 서버 시작 시, 이미 상태가 맞는 job들 처리
        asyncio.create_task(self._initial_scan())

        # LISTEN 루프 진입
        await self._listen_loop()

    async def stop(self) -> None:
        """리스너 중지 (태스크/연결 정리)"""
        self.running = False
        logger.info("YE Job State Listener 중지 요청")

        # 실행 중 태스크 정리
        if self.pending_tasks:
            logger.info(f"실행 중 태스크 {len(self.pending_tasks)}개 완료 대기...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.pending_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("일부 태스크가 30초 내에 끝나지 않아 강제 종료")

        # DB 연결 종료
        if self.conn:
            try:
                await self.conn.close()
            finally:
                self.conn = None
                logger.info("YE Listener PostgreSQL 연결 종료")

    # --------------------------------------------------
    # 서버 시작 시 1회 초기 스캔
    # --------------------------------------------------
    async def _initial_scan(self) -> None:
        """
        서버 시작 시 한 번만 실행:
        - 이미 DB 상에서 상태가 맞는 job들 (user_img_input/gen_vlm_analyze, done)을 찾아서 처리
        """
        asyncpg_url = DATABASE_URL.replace("postgresql://", "postgres://")

        try:
            conn = await asyncpg.connect(asyncpg_url)
            try:
                rows = await conn.fetch(
                    """
                    SELECT job_id::text AS job_id,
                           tenant_id,
                           current_step,
                           status
                    FROM jobs
                    WHERE (current_step = 'user_img_input'  AND status = 'done')
                       OR (current_step = 'gen_vlm_analyze' AND status = 'done')
                    """
                )

                if rows:
                    logger.info(
                        f"[YE INITIAL SCAN] 조건에 맞는 job {len(rows)}개"
                    )

                for r in rows:
                    job_id = r["job_id"]
                    current_step = r["current_step"]
                    status = r["status"]
                    tenant_id = r["tenant_id"]

                    task = asyncio.create_task(
                        self._process_job_state_change(
                            job_id=job_id,
                            current_step=current_step,
                            status=status,
                            tenant_id=tenant_id,
                        )
                    )
                    self.pending_tasks.add(task)
                    task.add_done_callback(self.pending_tasks.discard)

            finally:
                await conn.close()

        except Exception as e:
            logger.error(
                f"[YE INITIAL SCAN] 초기 스캔 중 오류 (무시하고 진행): {e}",
                exc_info=True,
            )

    # --------------------------------------------------
    # LISTEN / NOTIFY 루프
    # --------------------------------------------------
    async def _listen_loop(self) -> None:
        """PostgreSQL LISTEN 루프 (끊기면 재연결)"""
        asyncpg_url = DATABASE_URL.replace("postgresql://", "postgres://")

        while self.running:
            try:
                self.conn = await asyncpg.connect(asyncpg_url)
                logger.info("YE Listener: PostgreSQL 연결 성공")

                await self.conn.add_listener(
                    "job_state_changed", self._handle_notification
                )
                logger.info("YE Listener: LISTEN 'job_state_changed' 시작")

                # 연결 유지를 위해 단순 sleep 루프
                while self.running:
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("YE Listener: listen 루프 취소")
                break
            except Exception as e:
                logger.error(f"YE Listener: listen 루프 오류: {e}", exc_info=True)
                if self.running:
                    logger.info(f"{self.reconnect_delay}초 후 재연결 시도...")
                    await asyncio.sleep(self.reconnect_delay)
            finally:
                if self.conn:
                    try:
                        await self.conn.remove_listener(
                            "job_state_changed", self._handle_notification
                        )
                    except Exception:
                        pass
                    try:
                        await self.conn.close()
                    except Exception:
                        pass
                    self.conn = None
                    logger.info("YE Listener: PostgreSQL 연결 정리 완료")

    # --------------------------------------------------
    # NOTIFY 이벤트 핸들러
    # --------------------------------------------------
    def _handle_notification(self, conn, pid, channel, payload: str) -> None:
        """NOTIFY 콜백 (동기 → 비동기 태스크로 위임)"""
        try:
            data = json.loads(payload)
            job_id = data.get("job_id")
            current_step = data.get("current_step")
            status = data.get("status")
            tenant_id = data.get("tenant_id")

            print(
                f"[YE LISTENER] NOTIFY 수신: job_id={job_id}, "
                f"current_step={current_step}, status={status}"
            )
            logger.info(
                f"[YE LISTENER] NOTIFY 수신: job_id={job_id}, "
                f"current_step={current_step}, status={status}, tenant_id={tenant_id}"
            )

            if not job_id:
                return

            task = asyncio.create_task(
                self._process_job_state_change(
                    job_id=job_id,
                    current_step=current_step,
                    status=status,
                    tenant_id=tenant_id,
                )
            )
            self.pending_tasks.add(task)
            task.add_done_callback(self.pending_tasks.discard)

        except Exception as e:
            logger.error(f"YE Listener: NOTIFY 처리 오류: {e}", exc_info=True)

    # --------------------------------------------------
    # 상태에 따라 실제 작업 분기 (파이프라인 트리거 호출)
    # --------------------------------------------------
    async def _process_job_state_change(
        self,
        job_id: str,
        current_step: Optional[str],
        status: Optional[str],
        tenant_id: Optional[str],
    ) -> None:
        """
        상태에 따라 실제 작업 분기:
        - user_img_input, done      -> /llava/analyze (gen_vlm_analyze)
        - gen_vlm_analyze, done     -> /generate     (img_gen)
        """

        if not current_step or not status:
            return

        # 🔑 "이벤트 단위"로 중복 방지
        event_key = f"{job_id}:{current_step}:{status}"

        # 이미 한 번 처리 완료된 이벤트면 스킵
        if event_key in self.processed_events:
            logger.debug(
                f"[YE WORKER] 이미 처리 완료된 이벤트 스킵: job_id={job_id}, "
                f"current_step={current_step}, status={status}"
            )
            return

        # 동시에 같은 이벤트가 처리 중이면 스킵
        if event_key in self.active_events:
            logger.debug(
                f"[YE WORKER] 이미 처리 중인 이벤트 스킵: job_id={job_id}, "
                f"current_step={current_step}, status={status}"
            )
            return

        # 처리 시작/완료 표시
        self.active_events.add(event_key)
        self.processed_events.add(event_key)

        try:
            await trigger_next_pipeline_stage_ye(
                job_id=job_id,
                current_step=current_step,
                status=status,
                tenant_id=tenant_id,
            )

        finally:
            self.active_events.discard(event_key)


# ------------------------------------------------------
# FastAPI main.py 에서 쓰는 전역 진입점
# ------------------------------------------------------

_listener: Optional[JobStateListenerYE] = None


async def start_listener() -> None:
    """
    main.lifespan 에서 호출되는 진입점
    """
    global _listener
    if _listener is not None:
        return

    _listener = JobStateListenerYE()
    # 무한 루프이기 때문에 백그라운드 태스크로 실행
    asyncio.create_task(_listener.start())
    logger.info("YE Job State Listener start_listener 호출됨")


async def stop_listener() -> None:
    """
    main.lifespan shutdown 시 호출
    """
    global _listener
    if _listener is None:
        return

    await _listener.stop()
    _listener = None
    logger.info("YE Job State Listener stop_listener 호출됨")
