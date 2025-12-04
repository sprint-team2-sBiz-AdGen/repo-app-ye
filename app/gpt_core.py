#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpt_core.py

- 누끼 딴 음식 이미지를 OpenAI GPT 비전 API로 분석해서
  배경 프롬프트에 쓸 subject 정보를 뽑아낸다.

출력 예시 (plan["subject"]):

{
  "decoration_main": "chestnuts",
  "decoration_sub": "mixed nuts",
  "bg_main_color": "warm beige",
  "bg_floor_color": "dark brown wood",
  "best_season": "autumn"
}
"""

import io
import json
import os
import gc
import re
import base64
from typing import Any, Dict, Optional

from PIL import Image
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# =========================================================
# OpenAI 클라이언트 (OPENAI_API_KEY 사용)
# =========================================================

_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    OpenAI 클라이언트를 lazy-load 로 초기화한다.
    OPENAI_API_KEY 는 환경변수에서 읽는다.
    """
    global _client
    if _client is None:
        _client = OpenAI()
        print("[gpt_core] OpenAI client initialized")
    return _client


# =========================================================
# 시스템 프롬프트 로딩 (DB + fallback)
#   - vlm_prompt_assets.prompt(JSONB) 에서 읽고, 없으면 기본 프롬프트 사용
# =========================================================
# DB에 설정된 값이 없을 때 사용할 기본 시스템 프롬프트
_DEFAULT_GPT_SYSTEM_PROMPT = """
You are a planner for a Korean FOOD ADVERTISEMENT BACKGROUND.

You will be given:
- A FOOD PHOTO of a single main dish (hero food), already cut out or masked.
- A STYLE KEY string (strategy_name) that indicates the background style.

The FOOD ITSELF is fixed and must NOT be changed.

Your job:

1) Decide the best season in KOREA for promoting this menu as a seasonal item.
2) Choose simple, concrete decoration objects that PBG can draw on the floor/table.
3) Choose background colors (wall and floor/table) that make the main food stand out.

──────────────── STEP 1 – best_season ────────────────
Choose exactly ONE value for "best_season" from:
  {spring, summer, autumn, winter, all_year}

Use these rules:

- Use "summer" for clearly cold menus:
  - bingsu, naengmyeon, cold noodles, iced drinks, cold desserts.

- Use "winter" for clearly hot/warming menus:
  - hot stews, hot pots, hot soups, tteokguk, kalguksu, hot porridge,
    winter-only warm dishes.

- Use "spring" for typical spring promotions:
  - strawberry desserts, cherry blossom themed cakes, light/fresh spring menus.

- Use "autumn" for typical autumn/harvest promotions:
  - chestnut or pumpkin desserts, roasted sweet potatoes, harvest-feeling menus.

- Use "all_year" when:
  - it is a normal menu that Koreans usually eat in any season
    without strong seasonal feeling,
  - and it is not clearly tied to a specific seasonal event.

If the FOOD itself very clearly looks like an event menu
(shape, decoration on the plate, printed text on the food, or obviously themed design),
map the event to a season and choose that season:

- Seollal / Lunar New Year → "winter"
- Chuseok / Korean Thanksgiving → "autumn"
- Christmas → "winter"
- Valentine’s Day → "winter"
- Pepero Day → "autumn"
- Halloween → "autumn"
- hot boknal samgyetang style → "summer"
- Black Day jjajangmyeon promotion → "spring"

If both a generic and an event-based season are possible, prefer the EVENT season.

──────────────── STEP 2 – decoration objects ────────────────
Pick SMALL decoration objects that can realistically be placed on the floor/table
around the hero food. They must NOT be a second main dish.

1. decoration_main
   - Short English noun phrase.
   - Usually plural when natural:
     - "chestnuts", "fresh strawberries", "maple leaves", "mini pumpkins".
   - It should match BOTH the food and the chosen best_season.
   - It should support the hero food, not compete with it.

2. decoration_sub
   - Short English noun phrase for a smaller, secondary decoration
     near decoration_main:
     - "mixed nuts", "mint leaves", "pine cones", "small berries", "gold beads".
   - If no suitable second decoration exists, use "none".

Preferred decoration categories:

- Fruits and slices:
  strawberries, blueberries, raspberries, lemon wedges, lime slices, orange segments
- Nuts and seeds:
  chestnuts, walnuts, almonds, mixed nuts, sunflower seeds, sesame seeds
- Herbs:
  rosemary sprigs, thyme sprigs, parsley leaves, mint leaves
- Spices and grains:
  peppercorns, chili flakes, star anise, cinnamon sticks, rice grains
- Small sweets:
  sugar cubes, chocolate pieces, cocoa powder, cookie crumbs
- Seasonal props (ONLY when clearly related to event/season):
  tiny Christmas ornaments, mini pumpkins, small pine cones, tiny red envelopes,
  small tassel charms, tiny heart confetti, pepero sticks, etc.
- Simple natural items:
  dried leaves, maple leaves, small branches, pine cones, small stones.

NEVER use:
- People, hands, faces, bodies, phones.
- Text, letters, logos, packaging, brand labels.
- Cutlery or tableware as decoration:
  fork, knife, spoon, chopsticks, plates, cups, bowls, glasses, mugs, bottles.
- Large objects that steal attention from the hero food.
- Vague phrases like "festive mood", "holiday scene" without a concrete object.

──────────────── STEP 3 – background colors ────────────────
3. bg_main_color
   - Short English color phrase (1–3 words) for the WALL / background.
   - Example: "warm beige", "deep green", "soft cream", "dark brown".
   - Choose a color that contrasts enough with the main food so the food stands out.
   - Avoid neon, extremely saturated unnatural colors.

4. bg_floor_color
   - Short English color phrase (1–3 words) for the FLOOR or TABLE.
   - Example: "dark brown wood", "light gray concrete", "natural wood brown".
   - It should harmonize with bg_main_color and keep the food as the main focus.

Global rules:

- Do NOT mention the food name, bowl, plate, chopsticks, or dishes directly
  in these fields.
- Do NOT include camera, lens, resolution, aspect ratio, or model names.
- Keep every value short and clean, directly usable in a diffusion prompt template.

──────────────── OUTPUT FORMAT (VERY IMPORTANT) ────────────────

You MUST respond ONLY with a single valid JSON object, with EXACTLY these keys
at the top level:

{
  "decoration_main": "<short english noun phrase>",
  "decoration_sub": "<short english noun phrase or 'none'>",
  "bg_main_color": "<short english color phrase>",
  "bg_floor_color": "<short english color phrase>",
  "best_season": "<one of: spring | summer | autumn | winter | all_year>"
}

- Do NOT add any other top-level keys.
- Do NOT add any extra nested objects.
- Do NOT wrap the JSON in markdown code fences.
- Do NOT output ```json or ``` at all.
""".strip()


def _extract_en(value: Any) -> str:
    """
    jsonb 컬럼에서 영어(en) 텍스트만 꺼내는 헬퍼.

    - dict: value["en"]
    - str : JSON 문자열이면 파싱 후 ["en"], 아니면 그대로 사용
    """
    if isinstance(value, dict):
        return str(value.get("en", "") or "").strip()

    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return str(obj.get("en", "") or "").strip()
        except Exception:
            return value.strip()

    return str(value or "").strip()


def load_gpt_system_prompt(
    db,
    uid: Optional[str] = None,
) -> str:
    """
    GPT VLM용 시스템 프롬프트를 vlm_prompt_assets 에서 읽어온다.

    - 현재는 uid 값은 사용하지 않고,
      prompt_type='gpt', prompt_version='v1' 조회
    - prompt 컬럼은 {"en": "..."} 형태의 JSONB
    - 조회에 실패하거나 값이 없으면 기본 프롬프트(_DEFAULT_GPT_SYSTEM_PROMPT)를 반환한다.
    """
    if db is None:
        print("[gpt_core] ⚠️ db is None, use default GPT-VLM system prompt")
        return _DEFAULT_GPT_SYSTEM_PROMPT

    ptype = "gpt"
    pver = "v1"

    try:
        row = (
            db.execute(
                text(
                    """
                    SELECT prompt
                    FROM "vlm_prompt_assets"
                    WHERE prompt_type = :ptype
                      AND prompt_version = :pver
                    LIMIT 1
                    """
                ),
                {"ptype": ptype, "pver": pver},
            )
            .mappings()
            .first()
        )
    except SQLAlchemyError as e:
        print(
            f"[gpt_core] ⚠ failed to load vlm_prompt_assets "
            f"(prompt_type={ptype}, prompt_version={pver}): {e}"
        )
        try:
            db.rollback()
        except Exception:
            pass
        return _DEFAULT_GPT_SYSTEM_PROMPT

    if row is None:
        print(
            f"[gpt_core] ⚠ vlm_prompt_assets not found "
            f"(prompt_type={ptype}, prompt_version={pver}), fallback to default."
        )
        return _DEFAULT_GPT_SYSTEM_PROMPT

    system_prompt = _extract_en(row["prompt"])
    if not system_prompt:
        print(
            f"[gpt_core] ⚠ empty system prompt in vlm_prompt_assets "
            f"(prompt_type={ptype}, prompt_version={pver}), fallback to default."
        )
        return _DEFAULT_GPT_SYSTEM_PROMPT

    print(
        "[gpt_core] loaded GPT-VLM system prompt from DB "
        f"(prompt_type={ptype}, prompt_version={pver}, chars={len(system_prompt)})"
    )
    return system_prompt


# =========================================================
# GPT JSON 파싱 유틸
# =========================================================

def _sanitize_json_candidate(s: str) -> str:
    """
    GPT 응답 안의 JSON 후보 문자열을 정리한다.

    - \\' 같은 단순 escape 를 복원
    - 유효하지 않은 escape 시퀀스는 '\' 만 제거해서 원문만 남긴다
    """
    s = s.replace("\\'", "'")
    s = re.sub(r'\\([^"\\/bfnrtu])', r'\1', s)
    return s


def _parse_vlm_plan_json(raw_text: str) -> Dict[str, Any]:
    """
    GPT 응답 전체 텍스트에서 JSON 객체만 추출해 dict 로 변환한다.

    처리 순서:
    1) ```json ... ``` 코드 블록을 우선 찾고, 없으면 ``` ... ``` 또는 전체 텍스트 사용
    2) 첫 '{' 부터 마지막 '}' 까지 블록만 떼어낸다
    3) 스마트 따옴표 등을 일반 따옴표로 치환
    4) 마지막 요소 뒤에 붙은 trailing comma 제거
    5) escape 정리 후 json.loads 로 파싱
    """
    if not raw_text:
        raise ValueError("VLM 응답이 비어 있습니다.")

    # 1) ```json ... ``` 코드 블록 우선
    m = re.search(r"```json(.*?)```", raw_text, re.S | re.I)
    if m:
        block = m.group(1)
    else:
        # 일반 ``` ... ``` 도 허용
        m = re.search(r"```(.*?)```", raw_text, re.S)
        if m:
            block = m.group(1)
        else:
            block = raw_text

    sanitized = block.strip()

    # 2) JSON 객체 블록만 추출 – 첫 '{' ~ 마지막 '}'
    m = re.search(r"\{.*\}", sanitized, re.S)
    if m:
        sanitized = m.group(0)

    # 3) 스마트 따옴표 등 치환
    sanitized = (
        sanitized
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    # 4) trailing comma 제거
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)

    # 5) escape / 공백 정리
    sanitized = sanitized.strip()
    sanitized = _sanitize_json_candidate(sanitized)

    try:
        return json.loads(sanitized)
    except json.JSONDecodeError as e:
        print("[gpt_core] ⚠ JSONDecodeError:", e)
        print("[gpt_core] ---- sanitized JSON candidate ----")
        print(sanitized)
        print("[gpt_core] ---------------------------------")
        raise


# =========================================================
# subject JSON 정규화
# =========================================================

_EXPECTED_SUBJECT_KEYS = {
    "decoration_main",
    "decoration_sub",
    "bg_main_color",
    "bg_floor_color",
    "best_season",
}


def _normalize_subject_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    GPT가 어떤 형태로 JSON을 반환하더라도,
    최종적으로 plan["subject"] 에 들어갈 dict 를 다음 5개 키로 정리한다.

      - decoration_main
      - decoration_sub
      - bg_main_color
      - bg_floor_color
      - best_season

    만약 JSON 구조가 { "subject": { ... } } 라면 안쪽 subject dict 를 사용하고,
    아니라면 최상위 dict 에서 바로 값을 찾는다.

    best_season 은 없으면 season → best_season 로 매핑을 시도하고,
    유효하지 않으면 "all_year" 로 떨어뜨린다.
    """
    # 1) 안쪽 subject 블록이 있으면 우선 사용
    if isinstance(obj.get("subject"), dict):
        raw_subj = obj["subject"]
    else:
        raw_subj = obj

    subject: Dict[str, Any] = {}

    # 2) best_season 처리 (season → best_season fallback)
    best_season = raw_subj.get("best_season")
    if not best_season and "season" in raw_subj:
        best_season = raw_subj.get("season")

    if isinstance(best_season, str):
        bs_clean = best_season.strip().lower()
    else:
        bs_clean = None

    allowed_seasons = {"spring", "summer", "autumn", "winter", "all_year"}
    if bs_clean not in allowed_seasons:
        bs_clean = "all_year"

    subject["best_season"] = bs_clean

    # 3) 나머지 필드 정리
    for key in ("decoration_main", "decoration_sub", "bg_main_color", "bg_floor_color"):
        val = raw_subj.get(key)
        if val is None:
            continue
        if not isinstance(val, str):
            val = str(val)
        val = val.strip()
        if not val:
            continue
        subject[key] = val

    return subject


# =========================================================
# GPT 비전 실행 로직
# =========================================================

def _run_gpt_vlm_analysis(
    image: Image.Image,
    strategy_name: str,
    db,
    prompt_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    누끼 딴 음식 이미지를 GPT 비전 API에 넣고,
    best_season / decoration_* / bg_* 필드를 포함한 JSON subject 정보를 얻는다.
    """
    client = get_openai_client()
    system_prompt = load_gpt_system_prompt(db=db, uid=prompt_uid)

    # 이미지를 PNG → base64 data URL 로 인코딩
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{img_b64}"

    # 스타일 키(strategy_name)도 컨텍스트 힌트로 전달
    user_text = (
        "You are analyzing a cut-out image of a single main food item for an ad.\n"
        f"The current background style key is '{strategy_name}'.\n"
        "Use the system instructions and return ONE JSON object with the required fields."
    )

    resp = client.chat.completions.create(
        model=os.getenv("GPT_VLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        max_tokens=512,
        temperature=0.0,
    )

    content = resp.choices[0].message.content

    if isinstance(content, list):
        # 새 SDK에서 content가 list 일 수 있음 → text 파트만 합쳐서 사용
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
            else:
                t = getattr(part, "text", None)
            if t:
                text_parts.append(t)
        raw_text = "".join(text_parts)
    else:
        raw_text = str(content)

    raw_obj = _parse_vlm_plan_json(raw_text)
    subject_plan = _normalize_subject_json(raw_obj)
    return subject_plan


# =========================================================
# 외부에서 호출하는 API
# =========================================================

def gpt_analyze_from_bytes(
    image_bytes: bytes,
    filename: str,
    strategy_name: str,
    width: int,
    height: int,
    pbg_prompt_version: Optional[str] = None,
    *,
    db=None,
    gpt_prompt_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    누끼/마스킹된 이미지를 bytes로 받아 GPT 비전 분석 결과를 기반으로
    plan 스켈레톤을 생성한다.

    반환되는 plan 구조:
    - image.filename      : 원본 파일 이름 (URL/asset_id 는 이후 단계에서 추가)
    - strategy_name       : 스타일 키 (tone_style_id 상응 값)
    - subject             : GPT 결과(JSON) – decoration_*, bg_*, best_season
    - meta.canvas_width   : 캔버스 폭
    - meta.canvas_height  : 캔버스 높이
    - meta.pbg_prompt_version
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    print(
        f"[gpt_core] analyze (GPT): filename={filename}, "
        f"pbg_prompt_version={repr(pbg_prompt_version)}, "
        f"strategy_name={strategy_name!r}"
    )

    subject = _run_gpt_vlm_analysis(
        image=img,
        strategy_name=strategy_name,
        db=db,
        prompt_uid=gpt_prompt_uid,
    )

    meta: Dict[str, Any] = {
        "canvas_width": width,
        "canvas_height": height,
    }
    if pbg_prompt_version:
        meta["pbg_prompt_version"] = pbg_prompt_version

    plan: Dict[str, Any] = {
        "image": {
            # 실제 업로드 URL / asset_id 는 main/pbg_core 쪽에서 채운다.
            "filename": filename,
        },
        "strategy_name": strategy_name,
        "subject": subject,
        "meta": meta,
    }

    return plan


def llava_analyze_from_bytes(
    image_bytes: bytes,
    filename: str,
    strategy_name: str,
    width: int,
    height: int,
    pbg_prompt_version: Optional[str] = None,
    *,
    db=None,
    llava_prompt_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    예전 LLAVA 이름을 사용하는 코드와의 호환을 위한 래퍼.

    내부적으로 gpt_analyze_from_bytes 를 그대로 호출한다.
    """
    return gpt_analyze_from_bytes(
        image_bytes=image_bytes,
        filename=filename,
        strategy_name=strategy_name,
        width=width,
        height=height,
        pbg_prompt_version=pbg_prompt_version,
        db=db,
        gpt_prompt_uid=llava_prompt_uid,
    )


# =========================================================
# 리소스 정리
# =========================================================

def release_gpt_model_and_processor() -> None:

    try:
        gc.collect()
        print("[gpt_core] release_gpt_model_and_processor(): nothing to release (GPT only)")
    except Exception as e:
        print(f"[gpt_core] WARNING: release_gpt_model_and_processor failed: {e}")
