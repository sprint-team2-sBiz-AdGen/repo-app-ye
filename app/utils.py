#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""이미지/JSON 파일 저장 및 경로 처리 유틸 모듈.

- 디스크에 이미지/JSON을 저장하고 URL ↔ 절대 경로를 변환한다.
- DB insert/update 는 담당하지 않는다 (이미지/plan 관련 DB 작업은 pbg_core/gen_store 에서 처리).
"""

import os
import uuid
import json
import datetime
from typing import Optional, Tuple, Dict, Any

from fastapi import HTTPException
from PIL import Image
from pathlib import Path

from app.config import ASSETS_DIR, PART_NAME


# ======================================================
# 기본 경로 유틸
# ======================================================

def abs_from_url(url: str) -> str:
    """
    /assets/로 시작하는 asset URL을 실제 파일 시스템 절대 경로로 변환한다.

    예:
        "/assets/ye/tenants/tenant01/..." ->
        f"{ASSETS_DIR}/ye/tenants/tenant01/..."
    """
    if not url.startswith("/assets/"):
        raise HTTPException(status_code=400, detail="asset_url must start with /assets/")
    return os.path.join(ASSETS_DIR, url[len("/assets/"):])


def _make_asset_dir(tenant_id: str, kind: str) -> Path:
    """
    저장할 asset 디렉터리를 생성하고 절대 경로를 반환한다.

    구조:
        {ASSETS_DIR}/{PART_NAME}/tenants/{tenant_id}/{kind}/{YYYY}/{MM}/{DD}
    """
    today = datetime.datetime.utcnow()
    rel_dir = Path(
        f"{PART_NAME}/tenants/{tenant_id}/{kind}/"
        f"{today.year}/{today.month:02d}/{today.day:02d}"
    )
    abs_dir = Path(ASSETS_DIR) / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    return abs_dir


# ======================================================
# 이미지 저장
# ======================================================

def save_asset(tenant_id: str, kind: str, image: Image.Image, ext: str = ".png") -> Dict[str, Any]:
    """
    이미지를 파일로 저장하고 기본 메타데이터를 반환한다.

    반환 값:
        {
            "asset_id": <UUID 문자열>,
            "url": "/assets/...",
            "width": <int>,
            "height": <int>,
        }
    """
    abs_dir = _make_asset_dir(tenant_id, kind)
    asset_id = str(uuid.uuid4())
    abs_path = abs_dir / f"{asset_id}{ext}"
    rel_path = abs_path.relative_to(ASSETS_DIR)

    image.save(abs_path)
    return {
        "asset_id": asset_id,
        "url": f"/assets/{rel_path.as_posix()}",
        "width": image.width,
        "height": image.height,
    }


# ======================================================
# JSON 저장 / 로드
# ======================================================

def save_json(tenant_id: str, kind: str, data: Dict[str, Any]) -> str:
    """
    JSON 데이터를 파일로 저장하고 해당 asset URL을 반환한다.

    반환 예:
        "/assets/{PART_NAME}/tenants/{tenant_id}/{kind}/YYYY/MM/DD/xxxx.json"
    """
    abs_dir = _make_asset_dir(tenant_id, kind)
    filename = f"{uuid.uuid4()}.json"
    abs_path = abs_dir / filename
    rel_path = abs_path.relative_to(ASSETS_DIR)

    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return f"/assets/{rel_path.as_posix()}"


def load_json(json_url: str) -> Dict[str, Any]:
    """
    save_json 으로 저장된 JSON 파일을 로드한다.

    json_url 은 "/assets/..." 형식이어야 한다.
    """
    abs_path = os.path.join(ASSETS_DIR, json_url.lstrip("/assets/"))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================
# 기타 유틸
# ======================================================

def parse_hex_rgba(
    s: Optional[str],
    default: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Tuple[int, int, int, int]:
    """
    16진수 RGBA 문자열을 (R, G, B, A) 튜플로 변환한다.

    지원 형식:
        - "RRGGBBAA" (8자리)
        - "RRGGBB"   (6자리, alpha=255 로 취급)

    잘못된 형식이거나 None 인 경우 default 를 반환한다.
    """
    if not s:
        return default

    s = s.strip().lstrip("#")
    if len(s) == 8:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        a = int(s[6:8], 16)
        return (r, g, b, a)

    if len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        a = 255
        return (r, g, b, a)

    return default
