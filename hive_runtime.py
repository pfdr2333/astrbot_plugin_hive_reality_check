from __future__ import annotations

import base64
import json
import logging
import math
import os
import struct
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import httpx

from .hive_free_api import (
    FreeApiError,
    normalize_free_audio_payload,
    normalize_free_video_payload,
)

_logger = logging.getLogger("astrbot_plugin_hive_reality_check")

HIVE_V3_ENDPOINT = (
    "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection"
)

# 免费接口整段上传视频的时长上限（扩展前端限制为 30 秒）。
FREE_VIDEO_MAX_SECONDS = 30

INTENT_KEYWORDS = (
    "核查事实",
    "检查真实性",
    "真实性",
    "真的假的",
    "真假",
    "真吗",
    "伪造",
    "合成",
    "深度伪造",
    "deepfake",
    "deep fake",
    "ai生成",
    "ai 生成",
    "ai做的",
    "是不是ai",
    "是否ai生成",
    "是ai生成的吗",
    "is ai generated",
    "ai-generated",
    "synthetic",
    "authenticity",
    "real or fake",
)

DIRECT_REQUIRED_CLASSES = ("ai_generated", "not_ai_generated", "deepfake")
DROP_CLASSES = {"inconclusive", "inconclusive_video"}
NON_ATTRIBUTION_CLASSES = {
    "ai_generated_audio",
    "not_ai_generated_audio",
}
VIDEO_AI_THRESHOLD = 0.9
VIDEO_DEEPFAKE_THRESHOLD = 0.5
MAX_SUSPICIOUS_TIMESTAMPS = 5
ROUND_DIGITS = 4

MEDIA_FORMAT_TO_MIME = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "m4v": "video/x-m4v",
}


class KvBackend(Protocol):
    async def get(self, key: str, default: Any) -> Any:
        ...

    async def put(self, key: str, value: Any) -> None:
        ...


class HiveClientProtocol(Protocol):
    async def analyze_media_bytes(
        self,
        media_bytes: bytes,
        *,
        content_type: str,
    ) -> dict[str, Any]:
        ...


class HiveFreeClientProtocol(Protocol):
    """免费 API 客户端能力：图片字节、整段视频字节与文本三类检测。"""

    async def analyze_media_bytes(
        self,
        media_bytes: bytes,
        *,
        content_type: str,
    ) -> dict[str, Any]:
        ...

    async def analyze_video_bytes(
        self,
        media_bytes: bytes,
        *,
        content_type: str,
    ) -> dict[str, Any]:
        ...

    async def analyze_text(self, text: str) -> dict[str, Any]:
        ...


class PaidQuotaExceededError(RuntimeError):
    """付费请求（或计入配额模式的任何请求）前懒校验发现额度不足。"""

    def __init__(self, kind: str, need: int) -> None:
        super().__init__(f"{kind} quota insufficient for {need} requests")
        self.kind = kind
        self.need = need


@dataclass(slots=True)
class PluginSettings:
    enabled: bool = True
    hive_secret_key: str = ""
    hive_v3_daily_request_limit: int = 20
    per_user_daily_limit: int = 10
    allow_video: bool = True
    result_min_score: float = 0.05
    request_timeout_seconds: int = 60
    hive_free_api_enabled: bool = True
    free_api_exempt_from_quota: bool = True
    require_authenticity_intent: bool = True


@dataclass(slots=True)
class MediaResolutionError:
    code: str
    message: str


@dataclass(slots=True)
class ResolvedMediaTarget:
    target_type: str
    source: str
    file_path: str
    media_format: str


@dataclass(slots=True)
class VideoFrameSample:
    timestamp_seconds: float
    jpeg_bytes: bytes


@dataclass(slots=True)
class VideoSamplingResult:
    estimated_samples: int
    samples: list[VideoFrameSample] = field(default_factory=list)
    failed_timestamps: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ImageAnalysis:
    ai_generated: float
    not_ai_generated: float
    deepfake: float
    top_sources: list[dict[str, Any]]
    c2pa: dict[str, Any] | None


def normalize_settings(raw: dict[str, Any] | None) -> PluginSettings:
    raw = raw or {}
    return PluginSettings(
        enabled=bool(raw.get("enabled", True)),
        hive_secret_key=str(raw.get("hive_secret_key", "") or ""),
        hive_v3_daily_request_limit=max(1, int(raw.get("hive_v3_daily_request_limit", 20))),
        per_user_daily_limit=max(0, int(raw.get("per_user_daily_limit", 10))),
        allow_video=bool(raw.get("allow_video", True)),
        result_min_score=max(0.0, min(1.0, float(raw.get("result_min_score", 0.05)))),
        request_timeout_seconds=max(1, int(raw.get("request_timeout_seconds", 60))),
        hive_free_api_enabled=bool(raw.get("hive_free_api_enabled", True)),
        free_api_exempt_from_quota=bool(raw.get("free_api_exempt_from_quota", True)),
        require_authenticity_intent=bool(raw.get("require_authenticity_intent", True)),
    )


def contains_authenticity_intent(text: str | None) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in INTENT_KEYWORDS)


def round_score(value: float | int | None) -> float:
    return round(float(value or 0.0), ROUND_DIGITS)


def normalize_media_mime(content_type: str) -> str:
    normalized = (content_type or "").strip().lower()
    if "/" in normalized:
        return normalized
    return MEDIA_FORMAT_TO_MIME.get(normalized, f"application/octet-stream")


def build_data_url(media_bytes: bytes, *, content_type: str) -> str:
    mime = normalize_media_mime(content_type)
    encoded = base64.b64encode(media_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_error_result(
    verdict: str,
    notes: Sequence[str],
    *,
    target_type: str | None = None,
    source: str = "none",
) -> dict[str, Any]:
    return {
        "target_type": target_type,
        "source": source,
        "quota_consumed": 0,
        "verdict": verdict,
        "ai_generated": None,
        "not_ai_generated": None,
        "deepfake": None,
        "top_sources": [],
        "notes": [note for note in notes if note],
    }


def detect_image_type(path: str) -> str:
    with open(path, "rb") as file:
        header = file.read(32)
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "apng" if is_apng_file(path) else "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def is_apng_file(path: str) -> bool:
    with open(path, "rb") as file:
        signature = file.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            return False
        while True:
            length_bytes = file.read(4)
            if len(length_bytes) != 4:
                return False
            chunk_length = struct.unpack(">I", length_bytes)[0]
            chunk_type = file.read(4)
            if len(chunk_type) != 4:
                return False
            if chunk_type == b"acTL":
                return True
            file.seek(chunk_length + 4, os.SEEK_CUR)
            if chunk_type in {b"IDAT", b"IEND"}:
                return False


def sniff_video_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".webm", ".m4v"}:
        return suffix.lstrip(".")
    return "video"


def estimate_video_sample_count(path: str) -> int:
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("无法打开视频文件。")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps
        else:
            frame_total = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                frame_total += 1
            if frame_total <= 0:
                raise ValueError("无法估算视频时长。")
            duration_seconds = max(1.0, float(frame_total))
        return max(1, math.ceil(duration_seconds))
    finally:
        capture.release()


def sample_video_frames(path: str) -> VideoSamplingResult:
    estimated_samples = estimate_video_sample_count(path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法打开视频文件进行抽帧。")
    samples: list[VideoFrameSample] = []
    failed_timestamps: list[float] = []
    try:
        for second in range(estimated_samples):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                failed_timestamps.append(float(second))
                continue
            encoded_ok, encoded = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                failed_timestamps.append(float(second))
                continue
            samples.append(
                VideoFrameSample(
                    timestamp_seconds=float(second),
                    jpeg_bytes=encoded.tobytes(),
                )
            )
    finally:
        capture.release()
    return VideoSamplingResult(
        estimated_samples=estimated_samples,
        samples=samples,
        failed_timestamps=failed_timestamps,
    )


PAID_VIDEO_FRAME_COUNT = 5


def sample_video_frames_uniform(
    path: str, *, frame_count: int = PAID_VIDEO_FRAME_COUNT
) -> VideoSamplingResult:
    """按固定帧数（默认 5 帧）在视频时长内均匀抽帧，供付费通道使用。

    与逐秒采样不同：无论视频长短，都只送检固定帧数，便于把每次视频检测的
    付费请求数控制在上限内（计入用户/Hive 配额）。
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法打开视频文件进行抽帧。")
    samples: list[VideoFrameSample] = []
    failed_timestamps: list[float] = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count_prop = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and frame_count_prop > 0:
            duration_seconds = frame_count_prop / fps
        else:
            total = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                total += 1
            if total <= 0:
                raise ValueError("无法估算视频时长。")
            duration_seconds = max(1.0, float(total))
        count = max(1, int(frame_count))
        for index in range(count):
            timestamp = (index + 0.5) * duration_seconds / count
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                failed_timestamps.append(round(timestamp, ROUND_DIGITS))
                continue
            encoded_ok, encoded = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                failed_timestamps.append(round(timestamp, ROUND_DIGITS))
                continue
            samples.append(
                VideoFrameSample(
                    timestamp_seconds=round(timestamp, ROUND_DIGITS),
                    jpeg_bytes=encoded.tobytes(),
                )
            )
    finally:
        capture.release()
    return VideoSamplingResult(
        estimated_samples=count,
        samples=samples,
        failed_timestamps=failed_timestamps,
    )


def summarize_c2pa(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        payload.get("c2pa"),
        payload.get("metadata", {}).get("c2pa") if isinstance(payload.get("metadata"), dict) else None,
    ]
    output = payload.get("output")
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            candidates.extend(
                [
                    first.get("c2pa"),
                    first.get("metadata", {}).get("c2pa")
                    if isinstance(first.get("metadata"), dict)
                    else None,
                    first.get("extra", {}).get("c2pa")
                    if isinstance(first.get("extra"), dict)
                    else None,
                ]
            )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            summary: dict[str, Any] = {"present": True}
            for key in ("valid", "verified", "issuer", "generator", "claim_generator"):
                if key in candidate and isinstance(candidate[key], (str, bool, int, float)):
                    summary[key] = candidate[key]
            assertions = candidate.get("assertions")
            if isinstance(assertions, list):
                summary["assertion_count"] = len(assertions)
            actions = candidate.get("actions")
            if isinstance(actions, list):
                summary["action_count"] = len(actions)
            if len(summary) == 1:
                summary["available"] = True
            return summary
    return None


def _extract_output_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = payload.get("output")
    if isinstance(output, list):
        return [item for item in output if isinstance(item, dict)]
    if isinstance(output, dict):
        return [output]
    return []


def _extract_classes(payload: dict[str, Any]) -> dict[str, float]:
    class_scores: dict[str, float] = {}
    items = _extract_output_items(payload)
    if not items and isinstance(payload.get("classes"), list):
        items = [payload]
    for item in items:
        classes = item.get("classes")
        if not isinstance(classes, list):
            continue
        for entry in classes:
            if not isinstance(entry, dict):
                continue
            name = entry.get("class")
            value = entry.get("value")
            if not isinstance(name, str):
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            previous = class_scores.get(name, 0.0)
            if score > previous:
                class_scores[name] = score
    return class_scores


def _collect_top_sources(
    class_scores: dict[str, float],
    *,
    min_score: float,
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, float]] = []
    none_score = class_scores.get("none", 0.0)
    for name, value in class_scores.items():
        if (
            name in DIRECT_REQUIRED_CLASSES
            or name in DROP_CLASSES
            or name in NON_ATTRIBUTION_CLASSES
            or name == "none"
        ):
            continue
        if value >= min_score:
            candidates.append((name, value))
    candidates.sort(key=lambda item: item[1], reverse=True)
    if candidates:
        return [
            {"class": name, "score": round_score(score)}
            for name, score in candidates[:3]
        ]
    if none_score > 0:
        return [{"class": "none", "score": round_score(none_score)}]
    return []


def summarize_image_payload(
    payload: dict[str, Any],
    *,
    min_score: float,
) -> ImageAnalysis:
    class_scores = _extract_classes(payload)
    return ImageAnalysis(
        ai_generated=round_score(class_scores.get("ai_generated", 0.0)),
        not_ai_generated=round_score(class_scores.get("not_ai_generated", 0.0)),
        deepfake=round_score(class_scores.get("deepfake", 0.0)),
        top_sources=_collect_top_sources(class_scores, min_score=min_score),
        c2pa=summarize_c2pa(payload),
    )


def build_image_verdict(analysis: ImageAnalysis) -> str:
    if analysis.ai_generated >= VIDEO_AI_THRESHOLD and analysis.deepfake >= VIDEO_DEEPFAKE_THRESHOLD:
        return "likely_ai_generated_and_possible_deepfake"
    if analysis.ai_generated >= VIDEO_AI_THRESHOLD:
        return "likely_ai_generated"
    if analysis.deepfake >= VIDEO_DEEPFAKE_THRESHOLD:
        return "possible_deepfake"
    if analysis.not_ai_generated >= 0.9 and analysis.ai_generated < 0.5:
        return "likely_authentic"
    return "uncertain"


def build_text_verdict(analysis: ImageAnalysis) -> str:
    """文本判定：免费文本接口只返回 ai_generated 单类概率，无 deepfake 维度。"""
    if analysis.ai_generated >= VIDEO_AI_THRESHOLD:
        return "likely_ai_generated_text"
    if analysis.ai_generated < 0.5:
        return "likely_human_text"
    return "uncertain_text"


def aggregate_video_results(
    frame_payloads: Sequence[tuple[float, dict[str, Any]]],
    *,
    min_score: float,
) -> dict[str, Any]:
    frame_analyses = [
        (timestamp, summarize_image_payload(payload, min_score=min_score))
        for timestamp, payload in frame_payloads
    ]
    if not frame_analyses:
        raise ValueError("没有可用的视频帧检测结果。")
    max_ai = max(item.ai_generated for _, item in frame_analyses)
    max_not_ai = max(item.not_ai_generated for _, item in frame_analyses)
    max_deepfake = max(item.deepfake for _, item in frame_analyses)
    ai_timestamps = [
        round_score(timestamp)
        for timestamp, item in frame_analyses
        if item.ai_generated >= VIDEO_AI_THRESHOLD
    ][:MAX_SUSPICIOUS_TIMESTAMPS]
    deepfake_hit_timestamps = [
        round_score(timestamp)
        for timestamp, item in frame_analyses
        if item.deepfake >= VIDEO_DEEPFAKE_THRESHOLD
    ]
    deepfake_continuous = False
    last_hit: float | None = None
    for timestamp, item in frame_analyses:
        if item.deepfake >= VIDEO_DEEPFAKE_THRESHOLD:
            if last_hit is not None and abs(timestamp - last_hit) <= 1.0:
                deepfake_continuous = True
                break
            last_hit = timestamp
        else:
            last_hit = None
    total_frames = len(frame_analyses)
    ai_suspected = any(item.ai_generated >= VIDEO_AI_THRESHOLD for _, item in frame_analyses)
    deepfake_suspected = deepfake_continuous or (
        len(deepfake_hit_timestamps) / max(1, total_frames) >= 0.05
    )
    if ai_suspected and deepfake_suspected:
        verdict = "likely_ai_generated_and_possible_deepfake_video"
    elif ai_suspected:
        verdict = "likely_ai_generated_video"
    elif deepfake_suspected:
        verdict = "possible_deepfake_video"
    elif max_not_ai >= 0.9 and max_ai < 0.5:
        verdict = "likely_authentic_video"
    else:
        verdict = "uncertain_video"
    source_scores: dict[str, float] = {}
    c2pa: dict[str, Any] | None = None
    for _, item in frame_analyses:
        if item.c2pa and not c2pa:
            c2pa = item.c2pa
        for source in item.top_sources:
            name = str(source.get("class"))
            score = float(source.get("score", 0.0))
            source_scores[name] = max(source_scores.get(name, 0.0), score)
    top_sources = [
        {"class": name, "score": round_score(score)}
        for name, score in sorted(
            source_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
    ]
    return {
        "verdict": verdict,
        "ai_generated": round_score(max_ai),
        "not_ai_generated": round_score(max_not_ai),
        "deepfake": round_score(max_deepfake),
        "top_sources": top_sources,
        "frames_analyzed": total_frames,
        "suspicious_timestamps": {
            "ai_generated": ai_timestamps,
            "deepfake": deepfake_hit_timestamps[:MAX_SUSPICIOUS_TIMESTAMPS],
        },
        "ai_generated_video_suspected": ai_suspected,
        "deepfake_video_suspected": deepfake_suspected,
        "c2pa": c2pa,
    }


class HiveV3Client:
    def __init__(
        self,
        *,
        secret_key: str,
        timeout_seconds: int,
        endpoint: str = HIVE_V3_ENDPOINT,
    ) -> None:
        self.secret_key = secret_key
        self.timeout_seconds = timeout_seconds
        self.endpoint = endpoint

    async def analyze_media_bytes(
        self,
        media_bytes: bytes,
        *,
        content_type: str,
    ) -> dict[str, Any]:
        payload = {
            "media_metadata": True,
            "input": [
                {
                    "media_base64": build_data_url(
                        media_bytes,
                        content_type=content_type,
                    ),
                }
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.endpoint, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _extract_hive_error_text(response)
            if detail:
                raise RuntimeError(
                    f"Hive API request failed with HTTP {response.status_code}: {detail}"
                ) from exc
            raise RuntimeError(
                f"Hive API request failed with HTTP {response.status_code}."
            ) from exc
        return response.json()


def _extract_hive_error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        text = response.text.strip()
        return text[:500]
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        errors = payload.get("errors")
        if isinstance(errors, list):
            parts = []
            for item in errors:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    item_text = item.get("message") or item.get("detail") or item.get("error")
                    if isinstance(item_text, str) and item_text.strip():
                        parts.append(item_text.strip())
            if parts:
                return "; ".join(parts[:3])
    return ""


class QuotaManager:
    def __init__(
        self,
        backend: KvBackend,
        *,
        date_provider: Callable[[], date] | None = None,
    ) -> None:
        self.backend = backend
        self.date_provider = date_provider or (lambda: datetime.now().date())

    def _today(self) -> str:
        return self.date_provider().isoformat()

    def user_key(self, platform_id: str, sender_id: str) -> str:
        return f"quota:user:{self._today()}:{platform_id}:{sender_id}"

    def hive_key(self) -> str:
        return f"quota:hive_v3:{self._today()}"

    async def get_user_used(self, platform_id: str, sender_id: str) -> int:
        return int(await self.backend.get(self.user_key(platform_id, sender_id), 0) or 0)

    async def get_hive_used(self) -> int:
        return int(await self.backend.get(self.hive_key(), 0) or 0)

    async def increment_user(self, platform_id: str, sender_id: str, amount: int) -> None:
        key = self.user_key(platform_id, sender_id)
        current = int(await self.backend.get(key, 0) or 0)
        await self.backend.put(key, current + amount)

    async def increment_hive(self, amount: int) -> None:
        key = self.hive_key()
        current = int(await self.backend.get(key, 0) or 0)
        await self.backend.put(key, current + amount)


class PluginKvBackend:
    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin

    async def get(self, key: str, default: Any) -> Any:
        return await self.plugin.get_kv_data(key, default)

    async def put(self, key: str, value: Any) -> None:
        await self.plugin.put_kv_data(key, value)


class MediaResolver:
    def __init__(self, image_type: type, video_type: type, reply_type: type) -> None:
        self.image_type = image_type
        self.video_type = video_type
        self.reply_type = reply_type

    async def resolve_target(
        self,
        message_chain: Sequence[Any],
        *,
        allow_video: bool,
    ) -> ResolvedMediaTarget | MediaResolutionError:
        direct_result = await self._find_supported_target(
            message_chain,
            source="current_message",
            allow_video=allow_video,
        )
        if isinstance(direct_result, ResolvedMediaTarget):
            return direct_result

        first_reply = next(
            (segment for segment in message_chain if isinstance(segment, self.reply_type)),
            None,
        )
        if first_reply is None:
            return direct_result
        reply_chain = getattr(first_reply, "chain", None)
        if not reply_chain:
            if direct_result.code != "no_media":
                return direct_result
            return MediaResolutionError(
                code="reply_chain_missing",
                message="平台未提供被引用媒体内容，请让用户直接发送媒体。",
            )
        reply_result = await self._find_supported_target(
            reply_chain,
            source="quoted_message",
            allow_video=allow_video,
        )
        if isinstance(reply_result, ResolvedMediaTarget):
            return reply_result
        if direct_result.code != "no_media":
            return direct_result
        return reply_result

    async def _find_supported_target(
        self,
        chain: Sequence[Any],
        *,
        source: str,
        allow_video: bool,
    ) -> ResolvedMediaTarget | MediaResolutionError:
        had_candidate = False
        last_error: MediaResolutionError | None = None
        for segment in chain:
            if isinstance(segment, self.reply_type):
                continue
            if isinstance(segment, self.image_type):
                had_candidate = True
                file_path = await segment.convert_to_file_path()
                image_type = detect_image_type(file_path)
                if image_type == "gif":
                    last_error = MediaResolutionError(
                        code="gif_not_supported",
                        message="不支持核查 GIF 动图，请发送静态图片。",
                    )
                    continue
                if image_type == "apng":
                    last_error = MediaResolutionError(
                        code="apng_not_supported",
                        message="不支持核查 APNG 动图，请发送静态 PNG 或其他静态图片。",
                    )
                    continue
                if image_type not in {"jpeg", "png", "webp"}:
                    last_error = MediaResolutionError(
                        code="unsupported_image_type",
                        message="仅支持静态 jpg/jpeg/png/webp 图片核查。",
                    )
                    continue
                return ResolvedMediaTarget(
                    target_type="image",
                    source=source,
                    file_path=file_path,
                    media_format=image_type,
                )
            if isinstance(segment, self.video_type):
                had_candidate = True
                if not allow_video:
                    last_error = MediaResolutionError(
                        code="video_disabled",
                        message="当前插件配置已禁用视频核查。",
                    )
                    continue
                file_path = await segment.convert_to_file_path()
                return ResolvedMediaTarget(
                    target_type="video",
                    source=source,
                    file_path=file_path,
                    media_format=sniff_video_format(file_path),
                )
        if last_error:
            return last_error
        if had_candidate:
            return MediaResolutionError(
                code="unsupported_media",
                message="消息中存在媒体，但没有可核查的静态图片或允许的视频。",
            )
        return MediaResolutionError(
            code="no_media",
            message="当前消息及其引用消息中未找到可核查媒体。",
        )


class RealityCheckService:
    def __init__(
        self,
        *,
        settings: PluginSettings,
        quota_manager: QuotaManager,
        resolver: MediaResolver,
        hive_client_factory: Callable[[PluginSettings], HiveClientProtocol],
        free_client_factory: (
            Callable[[PluginSettings], HiveFreeClientProtocol | None] | None
        ) = None,
        video_estimator: Callable[[str], int] = estimate_video_sample_count,
        video_sampler: Callable[[str], VideoSamplingResult] = sample_video_frames_uniform,
    ) -> None:
        self.settings = settings
        self.quota_manager = quota_manager
        self.resolver = resolver
        self.hive_client_factory = hive_client_factory
        self.v3_client = hive_client_factory(settings)
        self.free_client = free_client_factory(settings) if free_client_factory else None
        self.video_estimator = video_estimator
        self.video_sampler = video_sampler

    def _has_key(self) -> bool:
        return bool(self.settings.hive_secret_key.strip())

    def _has_free(self) -> bool:
        return self.settings.hive_free_api_enabled and self.free_client is not None

    def _counts_toward_quota(self, *, backend: str) -> bool:
        """请求是否消耗配额：付费请求必计；免费请求仅当豁免关闭时计。"""
        return backend == "hive_v3" or not self.settings.free_api_exempt_from_quota

    async def _ensure_paid_quota(
        self,
        platform_id: str,
        sender_id: str,
        is_admin: bool,
        *,
        need: int,
    ) -> None:
        hive_used = await self.quota_manager.get_hive_used()
        hive_remaining = max(0, self.settings.hive_v3_daily_request_limit - hive_used)
        if hive_remaining < need:
            raise PaidQuotaExceededError("hive", need)
        if not is_admin:
            user_used = await self.quota_manager.get_user_used(platform_id, sender_id)
            user_remaining = max(0, self.settings.per_user_daily_limit - user_used)
            if user_remaining < need:
                raise PaidQuotaExceededError("user", need)

    async def _ensure_quota_or_error(
        self,
        platform_id: str,
        sender_id: str,
        is_admin: bool,
        *,
        need: int,
        target_type: str,
        source: str,
    ) -> dict[str, Any] | None:
        """额度足够返回 None，不足返回与现有文案一致的配额错误结果。"""
        try:
            await self._ensure_paid_quota(
                platform_id, sender_id, is_admin, need=need
            )
        except PaidQuotaExceededError as exc:
            return self._quota_error_result(
                exc, target_type=target_type, source=source
            )
        return None

    def _quota_error_result(
        self,
        exc: PaidQuotaExceededError,
        *,
        target_type: str,
        source: str,
    ) -> dict[str, Any]:
        frame_hint = (
            f"（视频按 {PAID_VIDEO_FRAME_COUNT} 帧均匀抽帧）" if target_type == "video" else ""
        )
        if exc.kind == "hive":
            limit = self.settings.hive_v3_daily_request_limit
            if exc.need <= 1:
                return build_error_result(
                    "hive_quota_exhausted",
                    [f"Hive 全局日额度已耗尽（上限 {limit} 次/天），请明天再试。"],
                    target_type=target_type, source=source,
                )
            return build_error_result(
                "hive_quota_insufficient",
                [
                    f"Hive 全局剩余额度不足（上限 {limit} 次/天），"
                    f"当前视频预计需要 {exc.need} 次请求{frame_hint}。"
                ],
                target_type=target_type, source=source,
            )
        limit = self.settings.per_user_daily_limit
        if exc.need <= 1:
            return build_error_result(
                "user_quota_exhausted",
                [f"你的今日核查额度已用完（上限 {limit} 次/天）。"],
                target_type=target_type, source=source,
            )
        return build_error_result(
            "user_quota_insufficient",
            [
                f"你的剩余额度不足（上限 {limit} 次/天），"
                f"当前视频预计需要 {exc.need} 次请求{frame_hint}。"
            ],
            target_type=target_type, source=source,
        )

    async def analyze_event(self, event: Any) -> dict[str, Any]:
        if not self.settings.enabled:
            return build_error_result("plugin_disabled", ["插件已禁用。"])
        if (
            self.settings.require_authenticity_intent
            and not contains_authenticity_intent(getattr(event, "message_str", ""))
        ):
            return build_error_result(
                "tool_should_not_be_called",
                [
                    "仅当用户明确询问真实性、是否 AI 生成、deepfake、真假或伪造时才应调用本工具。"
                ],
            )
        if not self._has_key() and not self._has_free():
            return build_error_result(
                "missing_hive_key",
                ["尚未配置 Hive Secret Key，且免费 API 未启用。"],
            )

        is_admin = bool(event.is_admin())
        platform_id = str(event.get_platform_id() or "unknown_platform")
        sender_id = str(event.get_sender_id() or "unknown_user")

        target = await self.resolver.resolve_target(
            event.get_messages(),
            allow_video=self.settings.allow_video,
        )
        if isinstance(target, MediaResolutionError):
            if target.code == "no_media":
                text_result = await self._analyze_text(
                    event, platform_id, sender_id, is_admin
                )
                if text_result is not None:
                    return text_result
            return build_error_result(target.code, [target.message])

        if target.target_type == "image":
            return await self._analyze_image(
                target, platform_id, sender_id, is_admin
            )
        return await self._analyze_video(
            target, platform_id, sender_id, is_admin
        )

    async def _analyze_image(
        self,
        target: ResolvedMediaTarget,
        platform_id: str,
        sender_id: str,
        is_admin: bool,
    ) -> dict[str, Any]:
        with open(target.file_path, "rb") as file:
            media_bytes = file.read()
        mime = normalize_media_mime(target.media_format)

        backend = "hive_v3"
        if self._has_free():
            if not self.settings.free_api_exempt_from_quota:
                blocked = await self._ensure_quota_or_error(
                    platform_id, sender_id, is_admin,
                    need=1, target_type="image", source=target.source,
                )
                if blocked:
                    return blocked
            try:
                payload = await self.free_client.analyze_media_bytes(
                    media_bytes, content_type=mime,
                )
                backend = "free"
            except Exception as exc:
                _logger.warning("免费图片 API 检测失败，准备回退到付费接口：%s", exc)
                if not self._has_key():
                    return build_error_result(
                        "free_api_failed",
                        [f"免费图片 API 检测失败：{exc}"],
                        target_type="image", source=target.source,
                    )

        if backend == "hive_v3":
            blocked = await self._ensure_quota_or_error(
                platform_id, sender_id, is_admin,
                need=1, target_type="image", source=target.source,
            )
            if blocked:
                return blocked
            try:
                payload = await self.v3_client.analyze_media_bytes(
                    media_bytes, content_type=target.media_format,
                )
            except Exception as exc:
                return build_error_result(
                    "api_request_failed",
                    [f"Hive 付费接口请求失败：{exc}"],
                    target_type="image", source=target.source,
                )

        if self._counts_toward_quota(backend=backend):
            await self.quota_manager.increment_hive(1)
            if not is_admin:
                await self.quota_manager.increment_user(platform_id, sender_id, 1)
        analysis = summarize_image_payload(
            payload, min_score=self.settings.result_min_score,
        )
        notes = [
            "图片阈值参考：ai_generated>=0.9 视为高度疑似 AI 生成；deepfake>=0.5 视为可能深度伪造。"
        ]
        if backend == "free":
            quota_hint = (
                "未消耗 Secret Key 额度"
                if self.settings.free_api_exempt_from_quota
                else "计入每日限额"
            )
            notes.insert(0, f"本次检测使用免费 API 通道（{quota_hint}）。")
        elif self._has_free():
            notes.insert(0, "免费 API 请求失败，已自动回退到 Secret Key 付费接口。")
        result = {
            "target_type": "image",
            "source": target.source,
            "quota_consumed": 1 if self._counts_toward_quota(backend=backend) else 0,
            "verdict": build_image_verdict(analysis),
            "ai_generated": analysis.ai_generated,
            "not_ai_generated": analysis.not_ai_generated,
            "deepfake": analysis.deepfake,
            "top_sources": analysis.top_sources,
            "notes": notes,
        }
        if analysis.c2pa:
            result["c2pa"] = analysis.c2pa
        return result

    async def _analyze_video(
        self,
        target: ResolvedMediaTarget,
        platform_id: str,
        sender_id: str,
        is_admin: bool,
    ) -> dict[str, Any]:
        free_available = self._has_free()
        estimated_seconds = self.video_estimator(target.file_path)

        if free_available and estimated_seconds <= FREE_VIDEO_MAX_SECONDS:
            with open(target.file_path, "rb") as file:
                media_bytes = file.read()
            mime = normalize_media_mime(target.media_format)
            if not self.settings.free_api_exempt_from_quota:
                blocked = await self._ensure_quota_or_error(
                    platform_id, sender_id, is_admin,
                    need=1, target_type="video", source=target.source,
                )
                if blocked:
                    return blocked
            try:
                raw_payload = await self.free_client.analyze_video_bytes(
                    media_bytes, content_type=mime,
                )
                frame_payloads = normalize_free_video_payload(raw_payload)
            except Exception as exc:
                _logger.warning("免费视频 API 检测失败，准备回退到付费接口：%s", exc)
                if not self._has_key():
                    return build_error_result(
                        "free_api_failed",
                        [f"免费视频 API 检测失败：{exc}"],
                        target_type="video", source=target.source,
                    )
            else:
                counted = self._counts_toward_quota(backend="free")
                if counted:
                    await self.quota_manager.increment_hive(1)
                    if not is_admin:
                        await self.quota_manager.increment_user(
                            platform_id, sender_id, 1
                        )
                aggregated = aggregate_video_results(
                    frame_payloads,
                    min_score=self.settings.result_min_score,
                )
                audio_scores = normalize_free_audio_payload(raw_payload)
                audio_analysis: dict[str, Any] | None = None
                if audio_scores:
                    # 实测 data.audio 段内类别名与视频通道一致（ai_generated/not_ai_generated），
                    # 同时兼容 *_audio 后缀变体，取两者最大值。
                    ai_audio = round_score(
                        max(
                            audio_scores.get("ai_generated_audio", 0.0),
                            audio_scores.get("ai_generated", 0.0),
                        )
                    )
                    not_ai_audio = round_score(
                        max(
                            audio_scores.get("not_ai_generated_audio", 0.0),
                            audio_scores.get("not_ai_generated", 0.0),
                        )
                    )
                    if ai_audio >= VIDEO_AI_THRESHOLD:
                        audio_verdict = "likely_ai_generated_audio"
                    elif not_ai_audio >= VIDEO_AI_THRESHOLD:
                        audio_verdict = "likely_authentic_audio"
                    else:
                        audio_verdict = "uncertain_audio"
                    audio_analysis = {
                        "ai_generated_audio": ai_audio,
                        "not_ai_generated_audio": not_ai_audio,
                        "verdict": audio_verdict,
                    }
                quota_hint = (
                    "未消耗 Secret Key 额度"
                    if self.settings.free_api_exempt_from_quota
                    else "计入每日限额"
                )
                notes: list[str] = [
                    f"本次视频使用免费 API 整段检测（{quota_hint}），接口按秒返回概率。",
                    "任一采样点 ai_generated>=0.9 视为疑似 AI 视频。",
                    "若 deepfake>=0.5 连续命中 2 个采样点，或命中占比>=5%，视为疑似 deepfake 视频。",
                ]
                if audio_analysis:
                    notes.append(
                        "音频通道（免费接口 data.audio）：该通道内 ai_generated>=0.9 "
                        "视为疑似 AI 合成音频，not_ai_generated>=0.9 视为更可能为真实音频。"
                    )
                result = {
                    "target_type": "video",
                    "source": target.source,
                    "quota_consumed": 1 if counted else 0,
                    "verdict": aggregated["verdict"],
                    "ai_generated": aggregated["ai_generated"],
                    "not_ai_generated": aggregated["not_ai_generated"],
                    "deepfake": aggregated["deepfake"],
                    "top_sources": aggregated["top_sources"],
                    "notes": notes,
                    "frames_analyzed": aggregated["frames_analyzed"],
                    "suspicious_timestamps": aggregated["suspicious_timestamps"],
                    "ai_generated_video_suspected": aggregated[
                        "ai_generated_video_suspected"
                    ],
                    "deepfake_video_suspected": aggregated[
                        "deepfake_video_suspected"
                    ],
                }
                if audio_analysis:
                    result["audio_analysis"] = audio_analysis
                if aggregated.get("c2pa"):
                    result["c2pa"] = aggregated["c2pa"]
                return result
        elif free_available:
            if not self._has_key():
                return build_error_result(
                    "video_too_long_for_free_api",
                    [
                        f"视频时长约 {estimated_seconds} 秒，超过免费接口 30 秒限制，且未配置 Secret Key 付费接口。"
                    ],
                    target_type="video", source=target.source,
                )
            _logger.info(
                "视频时长约 %s 秒，超过 30 秒，跳过免费整段检测，使用付费接口。",
                estimated_seconds,
            )

        blocked = await self._ensure_quota_or_error(
            platform_id, sender_id, is_admin,
            need=PAID_VIDEO_FRAME_COUNT, target_type="video", source=target.source,
        )
        if blocked:
            return blocked
        client = self.v3_client
        sampling = self.video_sampler(target.file_path)
        frame_payloads: list[tuple[float, dict[str, Any]]] = []
        request_failures: list[float] = []
        first_failure_reason: str | None = None
        for sample in sampling.samples:
            try:
                payload = await client.analyze_media_bytes(
                    sample.jpeg_bytes,
                    content_type="jpeg",
                )
            except Exception as exc:
                request_failures.append(sample.timestamp_seconds)
                if first_failure_reason is None:
                    first_failure_reason = f"{type(exc).__name__}: {exc}"
                continue
            frame_payloads.append((sample.timestamp_seconds, payload))
            await self.quota_manager.increment_hive(1)
            if not is_admin:
                await self.quota_manager.increment_user(platform_id, sender_id, 1)

        if not frame_payloads:
            failure_detail = (
                f"（首次失败原因：{first_failure_reason}）" if first_failure_reason else ""
            )
            return build_error_result(
                "video_analysis_failed",
                [
                    f"视频抽帧或送检未得到任何成功结果{failure_detail}，请让用户改发静态图或更短的视频。"
                ],
                target_type="video",
                source=target.source,
            )

        aggregated = aggregate_video_results(
            frame_payloads,
            min_score=self.settings.result_min_score,
        )
        notes: list[str] = [
            f"视频按 {PAID_VIDEO_FRAME_COUNT} 帧均匀抽帧送检；任一帧 ai_generated>=0.9 视为疑似 AI 视频。",
            "若 deepfake>=0.5 连续命中 2 帧，或命中帧占比>=5%，视为疑似 deepfake 视频。",
        ]
        if sampling.failed_timestamps:
            notes.append(
                f"有 {len(sampling.failed_timestamps)} 个抽样时间点抽帧失败，已自动跳过。"
            )
        if request_failures:
            notes.append(
                f"有 {len(request_failures)} 帧请求付费接口失败，结果为部分成功"
                + (f"（首次失败：{first_failure_reason}）。" if first_failure_reason else "。")
            )
        if free_available:
            notes.insert(0, "免费视频 API 失败或超时长，已回退到付费 5 帧检测。")
        result = {
            "target_type": "video",
            "source": target.source,
            "quota_consumed": len(frame_payloads),
            "verdict": aggregated["verdict"],
            "ai_generated": aggregated["ai_generated"],
            "not_ai_generated": aggregated["not_ai_generated"],
            "deepfake": aggregated["deepfake"],
            "top_sources": aggregated["top_sources"],
            "notes": notes,
            "frames_analyzed": aggregated["frames_analyzed"],
            "suspicious_timestamps": aggregated["suspicious_timestamps"],
            "ai_generated_video_suspected": aggregated["ai_generated_video_suspected"],
            "deepfake_video_suspected": aggregated["deepfake_video_suspected"],
        }
        if aggregated.get("c2pa"):
            result["c2pa"] = aggregated["c2pa"]
        return result

    async def _analyze_text(
        self,
        event: Any,
        platform_id: str,
        sender_id: str,
        is_admin: bool,
    ) -> dict[str, Any] | None:
        """消息无媒体时检测文本真实性；无文本返回 None 交由上层返回 no_media。"""
        text = (getattr(event, "message_str", "") or "").strip()
        if not text:
            return None
        if not self._has_free():
            return build_error_result(
                "text_detection_unavailable",
                ["文本检测依赖免费 API 通道，请启用 hive_free_api_enabled 配置。"],
                target_type="text", source="current_message",
            )
        if len(text) > 2048:
            return build_error_result(
                "text_too_long",
                [f"文本长度 {len(text)} 超过 2048 字符限制，请分段核查。"],
                target_type="text", source="current_message",
            )
        if not self.settings.free_api_exempt_from_quota:
            blocked = await self._ensure_quota_or_error(
                platform_id, sender_id, is_admin,
                need=1, target_type="text", source="current_message",
            )
            if blocked:
                return blocked
        try:
            payload = await self.free_client.analyze_text(text)
        except Exception as exc:
            return build_error_result(
                "free_api_failed",
                [f"免费文本 API 检测失败：{exc}"],
                target_type="text", source="current_message",
            )
        counted = self._counts_toward_quota(backend="free")
        if counted:
            await self.quota_manager.increment_hive(1)
            if not is_admin:
                await self.quota_manager.increment_user(platform_id, sender_id, 1)
        analysis = summarize_image_payload(
            payload, min_score=self.settings.result_min_score,
        )
        quota_hint = (
            "未消耗 Secret Key 额度"
            if self.settings.free_api_exempt_from_quota
            else "计入每日限额"
        )
        notes = [
            f"本次文本检测使用免费 API 通道（{quota_hint}）。",
            "文本阈值参考：ai_generated>=0.9 视为高度疑似 AI 生成；<0.5 视为更可能为人类撰写。",
        ]
        return {
            "target_type": "text",
            "source": "current_message",
            "quota_consumed": 1 if counted else 0,
            "verdict": build_text_verdict(analysis),
            "ai_generated": analysis.ai_generated,
            "not_ai_generated": analysis.not_ai_generated,
            "deepfake": analysis.deepfake,
            "top_sources": analysis.top_sources,
            "notes": notes,
        }


def build_llm_tool_guidance_text(
    *,
    enabled: bool,
    allow_video: bool,
    has_direct_image: bool,
    has_direct_video: bool,
    has_reply: bool,
    text_detection_enabled: bool,
    require_authenticity_intent: bool,
) -> str:
    availability_parts: list[str] = []
    if has_direct_image:
        availability_parts.append("当前消息包含图片候选")
    if has_direct_video:
        availability_parts.append("当前消息包含视频候选")
    if has_reply:
        availability_parts.append("当前消息包含引用消息")
    if not availability_parts:
        availability_parts.append("当前消息未显式看到可用媒体候选")
    video_text = "允许视频核查" if allow_video else "视频核查已禁用"
    enabled_text = "工具可用" if enabled else "工具已禁用"
    text_text = (
        "消息无媒体时也可核查纯文本是否 AI 生成（≤2048 字符，仅免费通道）"
        if text_detection_enabled
        else "文本核查未启用"
    )
    gate_text = (
        "仅当用户明确在问真实性、真假、是否 AI 生成、deepfake、伪造或合成时才可调用"
        if require_authenticity_intent
        else "只要用户消息包含媒体或文本即可直接调用核查"
    )
    return (
        "关于工具 check_media_authenticity_hive："
        f"{enabled_text}；{video_text}；{text_text}；{gate_text}；"
        "仅核查当前消息直接携带的媒体，或当前消息引用消息中的媒体；"
        "仅支持静态 jpg/jpeg/png/webp，以及启用后的视频 mp4/webm/m4v；"
        "不要用于 GIF、APNG、引用链为空、或一次多媒体批量核查；"
        "若用户一次发多个媒体，请提示逐个核查。"
        "明确告知用户ai置信度和最可能的来源"
        f" 当前可见状态：{'；'.join(availability_parts)}。"
    )
