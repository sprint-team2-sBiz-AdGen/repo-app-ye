#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pbg_core.py

PBG(Photo Background Generation) 코어 모듈.

- GPT VLM / LLaVA 등이 만든 plan 을 입력으로 받아
  - plan.subject + DB 정보로 프롬프트/배치 정보를 완성하고
  - 캔버스/마스크를 생성한 뒤
  - PBG 파이프라인을 실행하고
  - 결과를 image_assets / gen_variants / job_variants 에 기록한다.
"""

import io
import json
import os
import gc
import time
import uuid
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from compel import Compel
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from sqlalchemy import text
from transparent_background import Remover

from app.utils import load_json, abs_from_url, save_json
from app.config import ASSETS_DIR, ASSETS_URL_PREFIX, PART_NAME
from app.gen_store import (
    save_canvas_and_mask_assets,
    save_generated_asset,
    create_gen_variant,
    update_gen_run_status,
    finish_gen_run,
    create_job_variant,
    update_job_step,
)

load_dotenv()

# =========================================================
# 디바이스 / 하이퍼파라미터 설정
# =========================================================

def pick_device() -> torch.device:
    """
    사용할 디바이스를 결정한다.
    - 우선 CUDA GPU
    - 그다음 MPS
    - 마지막으로 CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
print(f"[pbg_core] device={DEVICE.type}, dtype={DTYPE}")

PBG_STEPS = int(os.getenv("PBG_STEPS", "20"))
PBG_CN_SCALE = float(os.getenv("PBG_CN_SCALE", "0.8"))

_pbg_pipeline: Optional[DiffusionPipeline] = None
_remover: Optional[Remover] = None
_compel: Optional[Compel] = None

# plan.json 에 seed 가 없을 때 사용하는 기본 정책
PBG_SEED_BASE = int(os.getenv("PBG_SEED_BASE", "50"))
PBG_SEED_STEP = int(os.getenv("PBG_SEED_STEP", "50"))

# =========================================================
# Compel 헬퍼 함수
# =========================================================

def _get_pbg_pipeline() -> DiffusionPipeline:
    """
    PBG DiffusionPipeline 을 필요할 때 한 번만 로드한다.
    """
    global _pbg_pipeline
    if _pbg_pipeline is None:
        model_path = os.getenv(
            "PBG_MODEL_PATH", "models_cache/diffusers/photo-background-generation"
        )
        print(f"[pbg_core] 모델 로드: {model_path}")
        _pbg_pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=model_path,
            local_files_only=True,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        _pbg_pipeline.enable_attention_slicing()
    return _pbg_pipeline


def _get_compel() -> Compel:
    """
    PBG 파이프라인의 tokenizer / text_encoder 를 사용하는 Compel 인스턴스를 반환한다.
    처음 한 번만 생성하고 이후에는 재사용한다.
    """
    global _compel
    pipe = _get_pbg_pipeline()

    if _compel is None:
        tokenizer = getattr(pipe, "tokenizer", None)
        text_encoder = getattr(pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            raise RuntimeError(
                "[pbg_core] PBG pipeline 에 tokenizer/text_encoder 가 없어 Compel 을 사용할 수 없습니다."
            )
        _compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)

    return _compel


def _encode_prompt_with_weights(
    prompt: str,
    negative_prompt: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compel 가중치 문법이 포함된 프롬프트를 임베딩으로 변환한다.
    - prompt: positive 프롬프트
    - negative_prompt: negative 프롬프트
    """
    compel = _get_compel()
    prompt_embeds = compel(prompt)
    negative_embeds = compel(negative_prompt) if negative_prompt else None

    # 항상 로그 출력 (환경변수 제어 제거)
    print("[pbg_core] [compel] prompt: ", repr(prompt))
    print(
        "[pbg_core] [compel] prompt_embeds shape:",
        tuple(prompt_embeds.shape),
        "dtype:",
        prompt_embeds.dtype,
        "device:",
        prompt_embeds.device,
    )
    if negative_embeds is not None:
        print("[pbg_core] [compel] negative_prompt:", repr(negative_prompt))
        print(
            "[pbg_core] [compel] negative_embeds shape:",
            tuple(negative_embeds.shape),
            "dtype:",
            negative_embeds.dtype,
            "device:",
            negative_embeds.device,
        )

    return prompt_embeds, negative_embeds


def strip_weights_for_debug(text: str) -> str:
    """
    디버그용 유틸.
    - (foo:1.3) → foo
    - (foo bar: 1.5) → foo bar
    처럼 가중치 문법만 제거하고 나머지 텍스트는 유지한다.
    (현재는 내부에서 자동 호출되지는 않지만, 필요 시 수동 디버깅용으로 남겨둠.)
    """
    if not text:
        return ""
    # (something:1.3) 패턴 지우고 안의 내용만 남기기
    text = re.sub(r"\(([^:()]+):\s*[0-9.]+\)", r"\1", text)
    # 남은 괄호 정리
    text = re.sub(r"[()]+", " ", text)
    # 공백 정리
    text = " ".join(text.split())
    return text


# =========================================================
# Remover (누끼) 헬퍼
# =========================================================

def get_remover() -> Remover:
    """
    transparent_background Remover 를 한 번만 로드해서 재사용한다.
    CUDA 가 가능하면 CUDA, 아니면 CPU 를 사용한다.
    (MPS 환경에서는 CPU 사용)
    """
    global _remover
    if _remover is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _remover = Remover(device=dev, jit=True)
        print(f"[pbg_core] Remover loaded on {dev}")
    return _remover


def run_remove_background(img: Image.Image) -> Image.Image:
    """
    RGB 이미지를 받아 soft alpha(type='map')를 적용한 RGBA cutout 으로 변환한다.
    - 업로드 즉시 만드는 분석용 cutout
    - PBG 단계에서 cutout 이 없을 때 fallback 으로도 사용
    """
    rgb = img.convert("RGB")
    remover = get_remover()
    alpha_map = remover.process(rgb, type="map").convert("L")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha_map)
    return rgba


# =========================================================
# 레이아웃 / 배치 헬퍼
# =========================================================

def ensure_rgba(im: Image.Image) -> Image.Image:
    """이미지를 RGBA 모드로 통일한다."""
    if im.mode == "RGBA":
        return im
    return im.convert("RGBA")


def crop_rgba_to_alpha_bbox(img: Image.Image, pad: int = 0) -> Image.Image:
    """
    alpha > 0 인 영역을 둘러싸는 박스를 기준으로 타이트하게 크롭한다.
    알파가 전부 0이면 원본 이미지를 그대로 반환한다.
    """
    img = ensure_rgba(img)
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return img

    x0, y0, x1, y1 = bbox
    if pad > 0:
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(img.width,  x1 + pad)
        y1 = min(img.height, y1 + pad)

    return img.crop((x0, y0, x1, y1))


def scale_to_long_by_percent(
    img: Image.Image,
    canvas_w: int,
    canvas_h: int,
    size_percent: float,
) -> Image.Image:
    """
    피사체의 긴 변을 기준으로 스케일을 조정한다.
    - 캔버스 긴 변 * size_percent(%) 에 맞추어 리사이즈
    """
    if canvas_w <= 0 or canvas_h <= 0:
        return img

    size_percent = max(1.0, float(size_percent))

    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    subject_long = max(w, h)
    canvas_long = max(canvas_w, canvas_h)

    target_long = int(canvas_long * (size_percent / 100.0))
    if target_long < 1:
        target_long = 1

    scale = target_long / float(subject_long)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return img.resize((new_w, new_h), Image.LANCZOS)


def place_subject_by_percent(
    canvas_w: int,
    canvas_h: int,
    subject_rgba: Image.Image,
    x_percent: float,
    y_percent: float,
) -> Tuple[Image.Image, Image.Image]:
    """
    피사체를 x/y 퍼센트 좌표에 배치한다.

    반환:
      - placed_rgb: 피사체가 올라간 RGB 캔버스
      - bg_mask   : inpaint 에 사용할 마스크 (255=배경, 0=피사체)
    """
    cx = int(canvas_w * (float(x_percent) / 100.0))
    cy = int(canvas_h * (float(y_percent) / 100.0))

    placed_rgba = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sw, sh = subject_rgba.size

    left = cx - sw // 2
    top = cy - sh // 2

    placed_rgba.alpha_composite(subject_rgba, dest=(left, top))

    fg_mask = placed_rgba.split()[-1]
    bg_mask = Image.new("L", (canvas_w, canvas_h), 255)
    bg_mask.paste(0, mask=fg_mask)

    placed_rgb = placed_rgba.convert("RGB")
    return placed_rgb, bg_mask


# =========================================================
# 스타일 이름 / 버전 정규화 + UUID 체크
# =========================================================

def normalize_strategy_name(name: str) -> str:
    """prompt_type 비교 전에 앞뒤 공백을 정리한다."""
    if not name:
        return ""
    return str(name).strip()


def _looks_like_uuid(value: str) -> bool:
    """문자열이 UUID 형식인지 대략 판별한다."""
    if not value:
        return False
    try:
        uuid.UUID(str(value))
        return True
    except Exception:
        return False


def _normalize_prompt_version(raw: Optional[str]) -> str:
    """
    pbg_prompt_version 보호용.
    - None / 공백 → 환경변수 또는 'v1'
    - UUID 처럼 생긴 문자열이 들어오면 잘못된 값으로 보고 기본값으로 강제.
    """
    raw_str = (raw or "").strip()
    if not raw_str:
        return os.getenv("PBG_PROMPT_VERSION", "v1")

    # UUID처럼 생기면 tenant_id 등이 잘못 들어온 값일 가능성이 큼
    if _looks_like_uuid(raw_str):
        print(
            f"[pbg_core] ⚠ prompt_version 이 UUID 처럼 보입니다({raw_str}) → "
            f"기본값(PBG_PROMPT_VERSION or 'v1')으로 강제합니다."
        )
        return os.getenv("PBG_PROMPT_VERSION", "v1")

    return raw_str


# =========================================================
# LLM subject 필드 정리
# =========================================================

def _clean_field(v: Optional[str]) -> Optional[str]:
    """
    빈 문자열이나 'uncertain' 같은 값은 None 으로 간주한다.
    이렇게 정리해 두면 프롬프트에서 자연스럽게 제외할 수 있다.
    """
    if not v:
        return None
    v = str(v).strip()
    if not v or v.lower() == "uncertain":
        return None
    return v


# =========================================================
# 프롬프트 유틸
# =========================================================

def _normalize_spaces(text: str) -> str:
    """
    공백/개행을 정리해 공백 하나 기준의 문자열로 만든다.
    """
    return " ".join(str(text).split())


# =========================================================
# 프롬프트 생성 (GPT subject + DB)
# =========================================================

def generate_background_variants(
    strategy_name: str,
    db,
    subject: Optional[Dict[str, Any]] = None,
    prompt_version: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    strategy_name 은 tone_style_id(UUID) 이거나 예전 prompt_type 문자열일 수 있다.

    조회 순서:
      1. tone_style_id 기준으로 pbg_prompt_assets 검색
      2. 없으면 prompt_type 문자열로 fallback

    반환 구조 예시:
      {
        "base_prompt": {
          "prompt": "...",
          "negative_prompt": "..."
        },
        "meta": {
          "prompt_type": "Hero Dish Focus",
          "tone_style_id": "<uuid 또는 None>"
        }
      }
    """
    strategy_raw = normalize_strategy_name(strategy_name)
    prompt_version = _normalize_prompt_version(prompt_version)

    tone_uuid: Optional[uuid.UUID] = None
    if strategy_raw:
        try:
            tone_uuid = uuid.UUID(strategy_raw)
        except Exception:
            tone_uuid = None

    row = None
    try:
        # 1) tone_style_id 기준 조회 (새 구조)
        if tone_uuid is not None:
            row = (
                db.execute(
                    text(
                        """
                        SELECT prompt, negative_prompt, prompt_type, tone_style_id
                        FROM "pbg_prompt_assets"
                        WHERE tone_style_id = :tone_id
                          AND prompt_version = :pver
                        LIMIT 1
                        """
                    ),
                    {"tone_id": tone_uuid, "pver": prompt_version},
                )
                .mappings()
                .first()
            )

        # 2) fallback: prompt_type 문자열 기반 조회 (예전 구조)
        if row is None:
            row = (
                db.execute(
                    text(
                        """
                        SELECT prompt, negative_prompt, prompt_type, tone_style_id
                        FROM "pbg_prompt_assets"
                        WHERE prompt_type = :ptype
                          AND prompt_version = :pver
                        LIMIT 1
                        """
                    ),
                    {"ptype": strategy_raw, "pver": prompt_version},
                )
                .mappings()
                .first()
            )

    except Exception as e:
        print(
            f'[pbg_core] ⚠ pbg_prompt_assets 조회 실패 '
            f'(strategy={strategy_raw}, version={prompt_version}): {e}'
        )
        row = None

    if row is None:
        raise ValueError(
            f'pbg_prompt_assets 에 tone_style_id="{strategy_raw}" 또는 '
            f'prompt_type="{strategy_raw}", prompt_version="{prompt_version}" 행이 없습니다.'
        )

    def _extract_en(value: Any) -> str:
        """prompt / negative 컬럼에서 en 필드를 꺼내는 헬퍼 함수."""
        if isinstance(value, dict):
            return str(value.get("en", "")).strip()
        if isinstance(value, str):
            try:
                obj = json.loads(value)
                if isinstance(obj, dict):
                    return str(obj.get("en", "")).strip()
            except Exception:
                return value.strip()
        return str(value).strip()

    base_db = _extract_en(row["prompt"])
    negative_db = _extract_en(row["negative_prompt"])

    if not base_db:
        raise ValueError(
            f"DB base prompt 가 비어 있습니다. "
            f"strategy={strategy_raw}, version={prompt_version}"
        )

    prompt_type_label = normalize_strategy_name(row.get("prompt_type") or strategy_raw)
    tone_style_id_str = None
    if row.get("tone_style_id") is not None:
        tone_style_id_str = str(row["tone_style_id"])
    elif tone_uuid is not None:
        tone_style_id_str = str(tone_uuid)

    # ---------- 2) subject 에서 per-image 힌트 뽑기 ----------
    subj = subject or {}

    best_season      = _clean_field(subj.get("best_season"))
    decoration_main  = _clean_field(subj.get("decoration_main"))
    decoration_sub   = _clean_field(subj.get("decoration_sub"))
    bg_main_color    = _clean_field(subj.get("bg_main_color"))
    bg_floor_color   = _clean_field(subj.get("bg_floor_color"))

    desc_parts = []

    # 계절 정보
    if best_season and best_season != "all_year":
        desc_parts.append(f"{best_season} seasonal mood")

    # 배경 색상
    if bg_main_color:
        desc_parts.append(f"{bg_main_color} wall in the background")
    if bg_floor_color:
        desc_parts.append(f"{bg_floor_color} floor or table surface")

    # 데코레이션
    if decoration_main:
        desc_parts.append(f"a small cluster of {decoration_main} near the main dish")
    if decoration_sub and decoration_sub.lower() != "none":
        desc_parts.append(f"a few {decoration_sub} close to the main decoration")

    subject_phrase = ", ".join(desc_parts)

    # ---------- 3) 최종 base_prompt 구성 ----------
    base_core = base_db.strip()
    if subject_phrase:
        base_core = base_core.rstrip(" ,.")
        prompt_full = f"{base_core}, {subject_phrase}"
    else:
        prompt_full = base_core

    prompt_full = _normalize_spaces(prompt_full)

    # ---------- 4) negative_prompt 구성 ----------
    neg_parts = []
    if negative_db:
        neg_parts.append(negative_db)

    extra_common = (
        "text, letters, korean letters, logo, watermark, caption, subtitle, ui, icon, "
        "people, faces, hands, phones, "
        "extra main dishes, extra plates, cups, spoons, forks, chopsticks, "
        "messy table, crowded buffet table"
    )
    neg_parts.append(extra_common)

    negative_full = _normalize_spaces(", ".join(neg_parts))

    return {
        "base_prompt": {
            "prompt": prompt_full,
            "negative_prompt": negative_full,
        },
        "meta": {
            "prompt_type": prompt_type_label,
            "tone_style_id": tone_style_id_str,
        },
    }


# =========================================================
# plan 에 background_prompts 붙이기
# =========================================================

def attach_background_prompts(
    plan: Dict[str, Any],
    db,
    tenant_id: str,
    prompt_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    plan 에 DB 기반 background_prompts 를 붙인다.

    사용 기준:
      - tone_style_id      : tone_styles.tone_style_id (UUID 또는 키 값)
      - tone_style_label   : 사람이 읽는 스타일 이름 (tone_styles.eng_name 등)
      - legacy strategy_name: 예전 스타일 키 (문자열) – label 로만 사용
    """

    # ---------- 0) tone_style_id / label / legacy strategy_name ----------
    tone_style_id: Optional[str] = None
    tone_style_label: Optional[str] = None

    # 새 구조: tone_style_id / tone_style_label
    if plan.get("tone_style_id"):
        tone_style_id = str(plan["tone_style_id"]).strip() or None

    if plan.get("tone_style_label"):
        tone_style_label = normalize_strategy_name(str(plan["tone_style_label"]))

    # 둘 다 없으면, 예전 필드 strategy_name 사용
    if not tone_style_id and not tone_style_label:
        legacy_name = plan.get("strategy_name")
        if legacy_name:
            tone_style_label = normalize_strategy_name(str(legacy_name))
            print(f"[pbg_core] tone_style_label 없음 → strategy_name 사용: {tone_style_label}")

    # 그래도 없으면 에러
    if not tone_style_id and not tone_style_label:
        raise ValueError(
            "plan 에 tone_style_id, tone_style_label, strategy_name 중 어떤 것도 없습니다. "
            "스타일 식별자가 필요합니다."
        )

    # DB 조회에 사용할 raw 키 (tone_style_id 우선, 없으면 label 사용)
    raw_strategy = tone_style_id or tone_style_label

    # ---------- 1) tone_styles 조회로 label 정리 ----------
    ts_row = None
    try:
        # UUID 형식처럼 보일 때만 tone_styles 에서 조회
        if raw_strategy and "-" in raw_strategy and len(raw_strategy) > 20:
            ts_row = (
                db.execute(
                    text(
                        """
                        SELECT tone_style_id, eng_name
                        FROM "tone_styles"
                        WHERE tone_style_id = :tid
                        LIMIT 1
                        """
                    ),
                    {"tid": raw_strategy},
                )
                .mappings()
                .first()
            )
    except Exception as e:
        print(f'[pbg_core] ⚠ tone_styles 조회 실패 (tone_style_id={raw_strategy}): {e}')
        ts_row = None

    if ts_row is not None:
        # tone_styles 에 있으면 그 값 사용
        tone_style_id = str(ts_row["tone_style_id"])
        tone_style_label = normalize_strategy_name(ts_row["eng_name"])
    else:
        # tone_styles 에 없으면 plan 안의 값을 정리해서 사용
        if tone_style_id is None:
            tone_style_id = raw_strategy
        if tone_style_label is None:
            tone_style_label = normalize_strategy_name(raw_strategy)

    if not tone_style_label:
        raise ValueError(
            f"tone_style_label 을 결정할 수 없습니다. raw_strategy={raw_strategy}"
        )

    # ---------- 2) subject / meta / prompt_version ----------
    subject = plan.get("subject") or {}

    # 예전 "season" 필드를 새 "best_season" 으로 매핑
    if "best_season" not in subject and "season" in subject:
        subject["best_season"] = subject["season"]

    plan["subject"] = subject

    meta = plan.get("meta") or {}

    # meta / 인자 / 환경변수 순으로 버전 결정 후 정규화
    if prompt_version:
        raw_pver = prompt_version
    else:
        raw_pver = meta.get("pbg_prompt_version") or os.getenv("PBG_PROMPT_VERSION", "v1")

    prompt_version = _normalize_prompt_version(raw_pver)

    # ---------- 3) base_prompt 생성 (subject + DB 템플릿) ----------
    # generate_background_variants 의 strategy_name 인자:
    #  - tone_style_id 가 있으면 그 값을 우선 사용
    #  - 없으면 tone_style_label 사용
    style_key_for_prompt = tone_style_id or tone_style_label

    variant_prompts = generate_background_variants(
        strategy_name=style_key_for_prompt,
        db=db,
        subject=subject,
        prompt_version=prompt_version,
    )

    base_cfg = variant_prompts.get("base_prompt")
    if base_cfg is None:
        raise ValueError(
            "generate_background_variants 가 base_prompt 를 반환하지 않았습니다. "
            f"style_key_for_prompt={style_key_for_prompt}"
        )

    # ---------- 4) placement preset 로드 ----------
    rows = (
        db.execute(
            text(
                """
                SELECT placement_preset_id, preset_order, x, y, size, rotation
                FROM "pbg_placement_presets"
                WHERE prompt_type = :ptype
                ORDER BY preset_order ASC
                """
            ),
            {"ptype": tone_style_label},
        )
        .mappings()
        .all()
    )

    if not rows:
        raise ValueError(
            f'pbg_placement_presets 에 prompt_type="{tone_style_label}" 행이 없습니다.'
        )

    # ---------- 5) 캔버스 크기/meta 기본값 ----------
    canvas_width = int(meta.get("canvas_width", 1080))
    canvas_height = int(meta.get("canvas_height", 1080))

    # ---------- 6) 0~1 → 0~100 퍼센트 변환 ----------
    def to_percent(v: Any, *, allow_zero: bool = False, default: float = 50.0) -> float:
        """
        DB 값이 0~1 범위이면 0~100 으로 스케일 업하고,
        이미 퍼센트 단위(0~100)에 가까우면 그대로 사용한다.
        """
        if v is None:
            return 0.0 if allow_zero else default

        try:
            val = float(v)
        except Exception:
            return default

        if 0.0 <= val <= 3.0:
            val = val * 100.0
        return val

    background_prompts: Dict[str, Any] = {}

    # ---------- 7) base_prompt ----------
    background_prompts["base_prompt"] = {
        "prompt": base_cfg["prompt"],
        "negative_prompt": base_cfg["negative_prompt"],
    }

    # ---------- 8) variant{n} : preset 별 배치 + seed ----------
    for r in rows:
        order = int(r["preset_order"])
        key = f"variant{order}"

        x_raw = r["x"]
        y_raw = r["y"]
        size_raw = r["size"]

        x_percent = to_percent(x_raw, allow_zero=True, default=50.0)
        y_percent = to_percent(y_raw, allow_zero=True, default=60.0)
        size_percent = to_percent(size_raw, allow_zero=False, default=80.0)

        # 범위 클램프
        x_percent = max(0.0, min(100.0, x_percent))
        y_percent = max(0.0, min(100.0, y_percent))
        size_percent = max(1.0, min(120.0, size_percent))

        rotation = float(r["rotation"] or 0.0)
        placement_preset_id = r.get("placement_preset_id") or r["placement_preset_id"]

        seed_value = PBG_SEED_BASE + (order - 1) * PBG_SEED_STEP

        background_prompts[key] = {
            "x_percent": x_percent,
            "y_percent": y_percent,
            "size_percent": size_percent,
            "rotation_deg": rotation,
            "placement_preset_id": str(placement_preset_id) if placement_preset_id else None,
            "seed": seed_value,
        }

    # ---------- 9) meta / plan 정리 ----------
    meta["canvas_width"] = canvas_width
    meta["canvas_height"] = canvas_height
    meta["pbg_prompt_version"] = prompt_version

    plan["meta"] = meta
    plan["background_prompts"] = background_prompts
    plan["tone_style_id"] = tone_style_id
    plan["tone_style_label"] = tone_style_label

    return plan


def finalize_plan_and_save(
    plan: Dict[str, Any],
    *,
    db,
    tenant_id: str,
    src_asset_id,
    cutout_asset_id,
    uploaded_url: str,
    cutout_url: str,
    run_id,
    canvas_width: int,
    canvas_height: int,
    pbg_prompt_version: str,
    creator_id=None,
) -> Dict[str, Any]:
    """
    GPT/LLaVA 가 만든 plan 스켈레톤에 다음 정보를 채워 넣고 저장한다.

    - 이미지 URL / asset_id
    - tenant_id / gen_run_id
    - meta (캔버스 사이즈, prompt 버전 등)
    - tone_style_id / tone_style_label
    - background_prompts

    그 후 plan.json 을 저장하고, 해당 URL 을 gen_runs.plan_json_url 에 기록한다.

    주의:
      - plan.json 내용 안에는 plan_json_url 을 넣지 않는다.
        (plan_json_url 은 gen_runs 테이블에서만 관리한다.)
    """

    # 1) meta 보정
    meta = plan.get("meta") or {}
    meta.setdefault("canvas_width", canvas_width)
    meta.setdefault("canvas_height", canvas_height)

    # pbg_prompt_version 정리 (이미 meta 에 있으면 그 값을 우선 정규화)
    if "pbg_prompt_version" in meta:
        meta["pbg_prompt_version"] = _normalize_prompt_version(meta["pbg_prompt_version"])
    elif pbg_prompt_version:
        meta["pbg_prompt_version"] = _normalize_prompt_version(pbg_prompt_version)
    else:
        meta["pbg_prompt_version"] = _normalize_prompt_version(None)

    plan["meta"] = meta

    # 2) image 블록 보정
    img_info = plan.get("image") or {}
    if "filename" not in img_info:
        img_info["filename"] = f"{src_asset_id}.png"

    img_info["uploaded_url"] = uploaded_url
    img_info["cutout_url"] = cutout_url
    img_info["src_asset_id"] = str(src_asset_id)
    img_info["cutout_asset_id"] = str(cutout_asset_id)

    if creator_id is not None:
        img_info["creator_id"] = str(creator_id)

    plan["image"] = img_info

    # 3) tenant / gen_run
    plan["tenant_id"] = tenant_id
    plan["gen_run_id"] = str(run_id)

    # job 기반 파이프라인이면 main 쪽에서 plan["job_id"] 를 채워 넣는다.

    # 4) background_prompts + tone_style_id / tone_style_label 붙이기
    plan = attach_background_prompts(
        plan=plan,
        db=db,
        tenant_id=tenant_id,
        prompt_version=plan["meta"].get("pbg_prompt_version"),
    )

    # 5) plan.json 저장 (현재 plan 에는 plan_json_url 이 없다)
    plan_json_url = save_json(tenant_id, "plan", plan)

    # 6) gen_runs 에 plan_json_url 기록
    try:
        db.execute(
            text(
                """
                UPDATE gen_runs
                SET plan_json_url = :plan_url
                WHERE run_id = :run_id
                """
            ),
            {
                "plan_url": plan_json_url,
                "run_id": run_id,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    return plan


# =========================================================
# PBG 배경 생성 (/generate)
# =========================================================

def pbg_background_to_png_bytes(
    plan_json: str,
    style_key: str,
    db,
    seed: Optional[int],
    tenant_id: str,
) -> bytes:
    """
    plan 정보를 기준으로 다음 단계를 실행한다.

    1. cutout/원본에서 피사체 RGBA 준비
    2. 배치 정보대로 캔버스/마스크 생성
    3. PBG 파이프라인 실행
    4. 결과/중간 산출물을 image_assets, gen_variants, job_variants 에 기록
    5. 최종 이미지를 PNG bytes 로 반환

    프롬프트:
      - background_prompts.base_prompt.prompt / negative_prompt 사용

    시드 선택:
      - background_prompts[style_key].seed 우선
      - 없으면 인자로 받은 seed
      - 둘 다 없으면 13 사용
    """
    if db is None:
        raise ValueError("DB 세션이 없습니다. PBG 생성 전에 DB 연결이 필요합니다.")

    overall_start = time.time()

    # plan_json 이 URL 인지, JSON string 인지 구분
    if plan_json.strip().startswith(ASSETS_URL_PREFIX):
        plan: Dict[str, Any] = load_json(plan_json)
    else:
        plan = json.loads(plan_json)

    # gen_run 상태를 running 으로 변경
    run_id_str = plan.get("gen_run_id")
    run_uuid: Optional[uuid.UUID] = None
    if run_id_str:
        try:
            run_uuid = uuid.UUID(str(run_id_str))
        except Exception:
            print(f"[pbg_core] ⚠ invalid gen_run_id={run_id_str}, ignore")

    if run_uuid is not None:
        try:
            update_gen_run_status(db, run_id=run_uuid, status="running")
        except Exception as e:
            # 이 쿼리가 실패하면 세션이 "aborted" 상태가 되기 때문에
            # 뒤에 다른 INSERT/SELECT 를 쓰려면 rollback 이 필수.
            print(f"[pbg_core] ⚠ update_gen_run_status(running) 실패: {e}")
            try:
                db.rollback()
            except Exception as rb_e:
                print(f"[pbg_core] ⚠ db.rollback() 도 실패: {rb_e}")

    # job_id 가 있으면 job_variants / jobs 상태 변경에 활용
    job_id_str = plan.get("job_id")
    job_uuid: Optional[uuid.UUID] = None
    if job_id_str:
        try:
            job_uuid = uuid.UUID(str(job_id_str))
        except Exception:
            print(f"[pbg_core] ⚠ invalid job_id={job_id_str}, ignore")

    # background_prompts 가 없거나 style_key / base_prompt 가 빠져 있으면 다시 채운다.
    bg_prompts = plan.get("background_prompts") or {}
    if style_key not in bg_prompts or "base_prompt" not in bg_prompts:
        meta = plan.get("meta") or {}
        prompt_version = meta.get("pbg_prompt_version") or None
        plan = attach_background_prompts(plan, db, tenant_id, prompt_version=prompt_version)
        bg_prompts = plan["background_prompts"]

    base_cfg = bg_prompts.get("base_prompt") or {}
    variant_cfg = bg_prompts.get(style_key)
    if not variant_cfg:
        raise ValueError(f"background_prompts['{style_key}'] 가 없습니다.")

    prompt_text: str = base_cfg.get(
        "prompt",
        "high quality commercial food photo, clean background",
    )
    negative_text: str = base_cfg.get(
        "negative_prompt",
        "blurry, text, watermark, logo, low quality, extra food, extra plates, cups",
    )

    # ---------------------------
    # subject RGBA 준비 (cutout 우선)
    # ---------------------------
    img_info = plan.get("image") or {}
    cutout_url = img_info.get("cutout_url")
    uploaded_url = img_info.get("uploaded_url")
    filename = img_info.get("filename")
    creator_id = img_info.get("creator_id")

    if cutout_url:
        src_path = abs_from_url(cutout_url)
        subject_rgba = Image.open(src_path).convert("RGBA")
    else:
        if uploaded_url:
            src_path = abs_from_url(uploaded_url)
        else:
            if not filename:
                raise ValueError("plan.image.cutout_url / uploaded_url / filename 모두 없습니다.")
            src_path = Path(ASSETS_DIR) / "shared" / f"{PART_NAME}-app" / "images" / filename

        src_rgb = Image.open(src_path).convert("RGB")
        subject_rgba = run_remove_background(src_rgb)

    subject_rgba = ensure_rgba(subject_rgba)
    subject_rgba = crop_rgba_to_alpha_bbox(subject_rgba, pad=4)

    # ---------------------------
    # 캔버스 / 배치
    # ---------------------------
    meta = plan.get("meta") or {}
    canvas_width = int(meta.get("canvas_width", 1080))
    canvas_height = int(meta.get("canvas_height", 1080))

    x_percent = float(variant_cfg.get("x_percent", 50.0))
    y_percent = float(variant_cfg.get("y_percent", 60.0))
    size_percent = float(variant_cfg.get("size_percent", 80.0))

    fg_scaled = scale_to_long_by_percent(
        subject_rgba,
        canvas_width,
        canvas_height,
        size_percent,
    )
    placed_rgb, bg_mask = place_subject_by_percent(
        canvas_width,
        canvas_height,
        fg_scaled,
        x_percent,
        y_percent,
    )

    tenant_id_final = plan.get("tenant_id") or tenant_id or "tenant01"

    # ----------------------------------------------------
    # 캔버스/마스크 저장 (image_assets)
    # ----------------------------------------------------
    canvas_meta, mask_meta = save_canvas_and_mask_assets(
        db=db,
        tenant_id=tenant_id_final,
        canvas_img=placed_rgb,
        mask_img=bg_mask,
        job_id=job_uuid,
        creator_id=creator_id,
    )
    canvas_asset_id = canvas_meta["asset_id"]
    mask_asset_id = mask_meta["asset_id"]

    # ---------------------------
    # PBG 실행
    # ---------------------------
    pipe = _get_pbg_pipeline()

    # 시드 선택: JSON → 인자 → 기본값
    seed_from_plan = variant_cfg.get("seed")
    if seed_from_plan is not None:
        seed_base = int(seed_from_plan)
    elif seed is not None:
        seed_base = int(seed)
    else:
        seed_base = 13

    try:
        generator = torch.Generator(device=DEVICE).manual_seed(seed_base)
    except Exception:
        generator = None

    print(f"[pbg_core] Generating style={style_key}, seed={seed_base}")

    # 가중치 프롬프트를 임베딩으로 변환
    prompt_embeds, negative_embeds = _encode_prompt_with_weights(
        prompt_text,
        negative_text,
    )

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            image=placed_rgb,
            mask_image=bg_mask,
            control_image=bg_mask,
            num_images_per_prompt=1,
            num_inference_steps=PBG_STEPS,
            guess_mode=False,
            controlnet_conditioning_scale=PBG_CN_SCALE,
            generator=generator,
        )
    infer_ms = (time.time() - t0) * 1000.0

    out_img: Image.Image = result.images[0]

    overall_latency_ms = (time.time() - overall_start) * 1000.0

    # ----------------------------------------------------
    # 생성 이미지 + gen_variants / job_variants 기록
    # ----------------------------------------------------
    gen_meta = save_generated_asset(
        db=db,
        tenant_id=tenant_id_final,
        generated_img=out_img,
        job_id=job_uuid,
        creator_id=creator_id,
    )
    bg_asset_id = gen_meta["asset_id"]

    placement_preset_id = variant_cfg.get("placement_preset_id")

    try:
        variant_index = int(style_key.replace("variant", ""))
    except Exception:
        variant_index = None

    if run_uuid is not None and variant_index is not None:
        try:
            # gen_variants 기록
            create_gen_variant(
                db=db,
                run_id=run_uuid,
                index=variant_index,
                canvas_asset_id=canvas_asset_id,
                mask_asset_id=mask_asset_id,
                bg_asset_id=bg_asset_id,
                placement_preset_id=placement_preset_id,
                prompt_en=prompt_text,
                negative_en=negative_text,
                seed_base=seed_base,
                steps=PBG_STEPS,
                infer_ms=infer_ms,
                latency_ms=overall_latency_ms,
            )
        except Exception as e:
            print(
                "[pbg_core] ⚠ create_gen_variant 실패 "
                f"(run_id={run_uuid}, style_key={style_key}, placement_preset_id={placement_preset_id}): {e}"
            )

        # job_variants 기록
        if job_uuid is not None and variant_index is not None:
            try:
                create_job_variant(
                    db=db,
                    job_id=job_uuid,
                    img_asset_id=bg_asset_id,
                    creation_order=variant_index,
                )
            except Exception as e:
                print(
                    f"[pbg_core] ⚠ create_job_variant 실패 "
                    f"(job_id={job_uuid}, creation_order={variant_index}): {e}"
                )

        # variant3 까지 생성이 끝나면 run 을 완료 상태로 변경한다.
        if style_key == "variant3":
            try:
                finish_gen_run(db=db, run_id=str(run_uuid), status="succeeded")
            except Exception as e:
                print(f"[pbg_core] ⚠ finish_gen_run 실패 (run_id={run_uuid}): {e}")

            # variant3 끝난 시점에 job 도 완료로 마킹
            if job_uuid is not None:
                try:
                    update_job_step(
                        db=db,
                        job_id=job_uuid,
                        current_step="img_gen",   # YE 단계 완료
                        status="done",
                    )
                except Exception as e:
                    print(f"[pbg_core] ⚠ update_job_step 실패(job 완료, job_id={job_uuid}): {e}")

    # PNG bytes 반환
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# =========================================================
# 리소스 정리
# =========================================================

def release_pbg_pipeline():
    """
    PBG DiffusionPipeline 을 메모리에서 내려 VRAM/RAM 을 비운다.
    서버에서 수동으로 언로드할 때 사용한다.
    """
    global _pbg_pipeline, _compel
    if _pbg_pipeline is not None:
        del _pbg_pipeline
        _pbg_pipeline = None
    if _compel is not None:
        del _compel
        _compel = None

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print("[pbg_core] ✅ PBG 파이프라인 언로드 완료")


def release_remover():
    """
    Remover 인스턴스를 정리해 메모리를 반환한다.
    """
    global _remover
    if _remover is not None:
        del _remover
        _remover = None
        gc.collect()
        print("[pbg_core] ✅ Remover 언로드 완료")
