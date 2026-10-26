from __future__ import annotations

from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.api import AstrBotConfig, FunctionTool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Reply, Video
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext

from .hive_free_api import HiveFreeClient
from .hive_runtime import (
    HiveV3Client,
    MediaResolver,
    PluginKvBackend,
    QuotaManager,
    RealityCheckService,
    build_llm_tool_guidance_text,
    compact_json,
    normalize_settings,
)


@dataclass
class HiveRealityCheckTool(FunctionTool[AstrAgentContext]):
    __pydantic_config__ = {"arbitrary_types_allowed": True}

    plugin: object = None
    name: str = "check_media_authenticity_hive"
    description: str = (
        "Check authenticity for one media item (or the current message text when no media is attached) "
        "in the current chat context using Hive detection (free API first, paid V3 fallback). "
        "Call this tool only when the user is explicitly asking whether a current-message or quoted-message image/video is authentic, AI-generated, synthetic, fake, or deepfake. "
        "This tool does not accept file paths or URLs. It reads only the current AstrMessageEvent context. "
        "Supported: static jpg/jpeg/png/webp, and video mp4/webm/m4v when video checking is enabled. "
        "Also supports plain-text AI detection (up to 2048 chars) when the current message has no media. "
        "Not supported: GIF, APNG, messages without media, empty quoted chains, and multi-media batch checks."
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        }
    )

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> str:
        del kwargs
        plugin = self.plugin
        if plugin is None:
            return compact_json(
                {
                    "target_type": None,
                    "source": "none",
                    "quota_consumed": 0,
                    "verdict": "plugin_runtime_missing",
                    "ai_generated": None,
                    "not_ai_generated": None,
                    "deepfake": None,
                    "top_sources": [],
                    "notes": ["插件运行时尚未初始化完成。"],
                }
            )
        result = await plugin.reality_service.analyze_event(context.context.event)
        return compact_json(result)


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.config = config
        self.settings = normalize_settings(dict(config))
        self.resolver = MediaResolver(Image, Video, Reply)
        self.quota_manager = QuotaManager(PluginKvBackend(self))
        self.reality_service = RealityCheckService(
            settings=self.settings,
            quota_manager=self.quota_manager,
            resolver=self.resolver,
            hive_client_factory=lambda settings: HiveV3Client(
                secret_key=settings.hive_secret_key,
                timeout_seconds=settings.request_timeout_seconds,
            ),
            free_client_factory=lambda settings: (
                HiveFreeClient(timeout_seconds=settings.request_timeout_seconds)
                if settings.hive_free_api_enabled
                else None
            ),
        )
        tool = HiveRealityCheckTool(plugin=self)
        tool.active = bool(self.settings.enabled)
        self.context.add_llm_tools(tool)
        logger.info("Hive reality check tool registered.")

    @filter.on_llm_request()
    async def decorate_llm_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        has_direct_image = False
        has_direct_video = False
        has_reply = False
        for segment in event.get_messages():
            if isinstance(segment, Image):
                has_direct_image = True
            elif isinstance(segment, Video):
                has_direct_video = True
            elif isinstance(segment, Reply):
                has_reply = True
        guidance = build_llm_tool_guidance_text(
            enabled=self.settings.enabled and (
                bool(self.settings.hive_secret_key.strip())
                or self.settings.hive_free_api_enabled
            ),
            allow_video=self.settings.allow_video,
            has_direct_image=has_direct_image,
            has_direct_video=has_direct_video,
            has_reply=has_reply,
            text_detection_enabled=self.settings.hive_free_api_enabled,
            require_authenticity_intent=self.settings.require_authenticity_intent,
        )
        req.system_prompt = f"{req.system_prompt or ''}\n{guidance}\n"
