"""Hive 免费检测 API 客户端（移植自 hive_detector.py）。

对应 Chrome 扩展 "Hive AI Detector"（v0.0.10，扩展 ID cmeikcgfecnhojcbfapbmpbjgllklcbi，
Base: https://plugin.hivemoderation.com）。按扩展 service_worker.js 的真实请求指纹复现：
- 相同的 Origin / Sec-Fetch-* / sec-ch-ua 头；referrerPolicy: no-referrer -> 不发送 Referer
- credentials: same-origin -> 不携带 Cookie
- multipart 字段顺序与扩展一致（media 在前，request_id 在后），filename 固定为 "blob"
- 文本请求体为紧凑 JSON（同 JSON.stringify），request_id 每次新生成 UUID v4

零第三方依赖（仅标准库）：同步 urllib 内核 + asyncio.to_thread 异步外壳，
避免阻塞插件所在的事件循环。归一化函数把免费响应转成与 Hive V3 相同的
{"output": [{"classes": [{"class", "value"}]}]} 形态，供插件现有汇总/判定逻辑直接复用。
"""

from __future__ import annotations

import asyncio
import json
import secrets
import string
import uuid
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

BASE_URL = "https://plugin.hivemoderation.com"
EXTENSION_ID = "cmeikcgfecnhojcbfapbmpbjgllklcbi"
TEXT_MAX_CHARS = 2048

# 模拟 Chrome service worker 发起跨域 fetch 时的浏览器指纹。
# 不设置 Referer（referrerPolicy: no-referrer）与 Cookie（跨域 same-origin）。
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": f"chrome-extension://{EXTENSION_ID}",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}


class FreeApiError(RuntimeError):
    """免费 API 请求失败（HTTP 错误、超时或响应结构不符合预期）。"""


def _free_error_text(payload: dict[str, Any]) -> str:
    for key in ("error", "message", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "响应缺少 data.classes 字段"


def normalize_free_classes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把免费响应 {"data": {"classes": [{"class", "score"}]}} 归一为 V3 形态。

    免费接口用 score 表示类别概率，V3 用 value；此处统一为 V3 结构，
    使 summarize_image_payload / aggregate_video_results 无需感知后端差异。
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        raise FreeApiError(_free_error_text(payload))
    classes = data.get("classes")
    if not isinstance(classes, list):
        raise FreeApiError(_free_error_text(payload))
    normalized: list[dict[str, Any]] = []
    for entry in classes:
        if not isinstance(entry, dict):
            continue
        name = entry.get("class")
        if not isinstance(name, str):
            continue
        normalized.append({"class": name, "value": entry.get("score")})
    if not normalized:
        raise FreeApiError(_free_error_text(payload))
    return {"output": [{"classes": normalized}]}


def normalize_free_video_payload(
    payload: dict[str, Any],
) -> list[tuple[float, dict[str, Any]]]:
    """把免费视频响应归一为 (时间点, V3 形态载荷) 序列。

    实测返回结构：{"data": {"duration": N, "video": [{"time": 0.0, "classes": [...]}, ...]}}，
    即免费接口已按每秒给出完整类别概率。归一化后直接喂给 aggregate_video_results，
    复用插件的逐帧取最大/连续 deepfake/时间戳等判定逻辑。
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        raise FreeApiError(_free_error_text(payload))
    video = data.get("video")
    if not isinstance(video, list) or not video:
        raise FreeApiError(_free_error_text(payload))
    frames: list[tuple[float, dict[str, Any]]] = []
    for item in video:
        if not isinstance(item, dict):
            continue
        try:
            timestamp = float(item.get("time", 0.0))
        except (TypeError, ValueError):
            timestamp = 0.0
        classes = item.get("classes")
        if not isinstance(classes, list):
            continue
        normalized: list[dict[str, Any]] = []
        for entry in classes:
            if not isinstance(entry, dict):
                continue
            name = entry.get("class")
            if not isinstance(name, str):
                continue
            normalized.append({"class": name, "value": entry.get("score")})
        if normalized:
            frames.append((timestamp, {"output": [{"classes": normalized}]}))
    if not frames:
        raise FreeApiError(_free_error_text(payload))
    return frames


def normalize_free_audio_payload(
    payload: dict[str, Any],
) -> dict[str, float] | None:
    """从免费视频响应提取音频类别最高分；无音频数据返回 None。

    目标结构（与 video 一致，实测待定稿）：{"data": {"audio": [{"time": ..., "classes": [...]}]}}。
    防御式解析：兼容 audio 缺省、直接为 classes 列表、条目无 time 等变体，
    返回 {class: max_score}，由调用方据此生成音频判定。
    """
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    audio = data.get("audio")
    if not isinstance(audio, list) or not audio:
        return None
    scores: dict[str, float] = {}
    for item in audio:
        if not isinstance(item, dict):
            continue
        classes = item.get("classes")
        if not isinstance(classes, list):
            classes = [item]
        for entry in classes:
            if not isinstance(entry, dict):
                continue
            name = entry.get("class")
            if not isinstance(name, str):
                continue
            try:
                value = float(entry.get("score"))
            except (TypeError, ValueError):
                continue
            scores[name] = max(scores.get(name, 0.0), value)
    return scores or None


class HiveFreeClient:
    """Hive 免费检测客户端：优先通道，失败时由调用方回退到付费 V3 接口。"""

    def __init__(self, *, timeout_seconds: int = 120) -> None:
        self.timeout_seconds = timeout_seconds

    # ---------- 内部工具（与 hive_detector.py 一致） ----------

    @staticmethod
    def _request_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _webkit_boundary() -> str:
        """生成 Chrome WebKit 风格的 multipart boundary。"""
        alphabet = string.ascii_letters + string.digits
        return "----WebKitFormBoundary" + "".join(
            secrets.choice(alphabet) for _ in range(16)
        )

    @staticmethod
    def _field(
        boundary: str,
        *,
        name: str,
        value: bytes | None = None,
        text: str | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> bytes:
        """构造单个 multipart 字段，media 走二进制分支、request_id 走纯文本分支。"""
        if value is None and text is not None:
            value = text.encode("utf-8")
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        chunk = [f"--{boundary}\r\n".encode(), disposition.encode(), b"\r\n"]
        if content_type:
            chunk.append(f"Content-Type: {content_type}\r\n".encode())
        chunk.append(b"\r\n")
        chunk.append(value or b"")
        chunk.append(b"\r\n")
        return b"".join(chunk)

    def _post(self, endpoint: str, body: bytes, content_type: str) -> dict[str, Any]:
        """发起 POST 并解析 JSON 响应；HTTP 错误统一转 FreeApiError。"""
        headers = dict(DEFAULT_HEADERS)
        headers["Content-Type"] = content_type
        req = urlrequest.Request(
            f"{BASE_URL}{endpoint}", data=body, headers=headers, method="POST"
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            raise FreeApiError(
                f"HTTP {exc.code} from {endpoint}: {exc.read().decode('utf-8', 'replace')}"
            ) from exc
        except urlerror.URLError as exc:
            raise FreeApiError(f"请求 {endpoint} 网络错误：{exc.reason}") from exc
        except OSError as exc:
            raise FreeApiError(f"请求 {endpoint} 连接失败：{exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise FreeApiError(f"请求 {endpoint} 响应解析失败：{exc}") from exc

    def _post_multipart(
        self,
        endpoint: str,
        *,
        media: bytes,
        mime: str,
        request_id: str,
    ) -> dict[str, Any]:
        """构造与扩展完全一致的 multipart/form-data 请求体（media 后接 request_id）。"""
        boundary = self._webkit_boundary()
        parts = [
            self._field(
                boundary, name="media", value=media,
                filename="blob", content_type=mime,
            ),
            self._field(boundary, name="request_id", text=request_id),
        ]
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        return self._post(
            endpoint, body, f"multipart/form-data; boundary={boundary}"
        )

    async def _post_async(
        self, endpoint: str, body: bytes, content_type: str
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._post, endpoint, body, content_type)

    async def _post_multipart_async(
        self,
        endpoint: str,
        *,
        media: bytes,
        mime: str,
        request_id: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._post_multipart, endpoint,
            media=media, mime=mime, request_id=request_id,
        )

    # ---------- 公开 API（返回已归一化为 V3 形态的载荷） ----------

    async def analyze_text(self, text: str) -> dict[str, Any]:
        """检测文本是否为 AI 生成（≤2048 字符）。"""
        if len(text) > TEXT_MAX_CHARS:
            raise ValueError(f"文本长度 {len(text)} 超过 {TEXT_MAX_CHARS} 限制")
        body = json.dumps(
            {"text": text, "request_id": self._request_id()},
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        payload = await self._post_async(
            "/api/v1/text/ai_detection", body, "application/json"
        )
        return normalize_free_classes_payload(payload)

    async def analyze_media_bytes(
        self, media_bytes: bytes, *, content_type: str
    ) -> dict[str, Any]:
        """检测图片字节（本地上传，filename 固定 "blob"）。"""
        payload = await self._post_multipart_async(
            "/api/v1/image/ai_detection",
            media=media_bytes, mime=content_type,
            request_id=self._request_id(),
        )
        return normalize_free_classes_payload(payload)

    async def analyze_video_bytes(
        self, media_bytes: bytes, *, content_type: str
    ) -> dict[str, Any]:
        """检测整段视频字节（免费接口限制 ≤30 秒），返回原始响应供逐时间点归一化。"""
        return await self._post_multipart_async(
            "/api/v1/video/ai_detection",
            media=media_bytes, mime=content_type,
            request_id=self._request_id(),
        )
