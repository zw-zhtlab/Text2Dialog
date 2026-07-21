#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件：统一管理对话链的所有配置
"""

import math
import os
from typing import Dict, List, Any

from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())


class ModelPlatform:
    """模型平台配置类"""

    # 支持的模型平台
    PLATFORMS: Dict[str, Dict[str, str]] = {
        "deepseek": {
            "api_key_env": "DEEPSEEK_API",
            "base_url_env": "DEEPSEEK_BASE_URL",
            "default_base_url": "https://api.deepseek.com",
            "default_model": "deepseek-chat",
            "description": "DeepSeek官方平台",
        },
         "siliconflow": {
            "api_key_env": "SILICONFLOW_API_KEY",
            "base_url_env": "SILICONFLOW_BASE_URL",
            "default_base_url": "https://api.siliconflow.cn/v1",
            "default_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "description": "硅基流动平台",
        },
        "bailian": {
            "api_key_env": "DASHSCOPE_API_KEY",
            "base_url_env": "BAILIAN_BASE_URL",
            "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "qwen-plus",
            "description": "阿里云百炼（通义千问）",
        },        
        "moonshot": {
            "api_key_env": "MOONSHOT_API_KEY",
            "base_url_env": "MOONSHOT_BASE_URL",
            "default_base_url": "https://api.moonshot.cn/v1",
            "default_model": "kimi-k2-0905-preview",
            "description": "月之暗面Kimi平台",
        },       
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
            "default_base_url": "https://api.openai.com/v1",
            "default_model": "gpt-5-mini",
            "description": "OpenAI官方平台",
        },
        "gemini": {
            "api_key_env": "GEMINI_API_KEY",
            "base_url_env": "GEMINI_BASE_URL",
            "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "default_model": "gemini-2.0-flash",
            "description": "谷歌Gemini官方平台",
        },
        "aws_bedrock": {
            "api_key_env": "AWS_BEDROCK_API_KEY",
            "base_url_env": "AWS_BEDROCK_BASE_URL",
            "default_base_url": "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
            "default_model": "openai.gpt-oss-20b-1:0",
            "description": "亚马逊Bedrock官方平台",
        },
        "custom": {
            "api_key_env": "CUSTOM_API_KEY",
            "base_url_env": "CUSTOM_BASE_URL",
            "default_base_url": "https://your-custom-endpoint.com/v1",
            "default_model": "custom-model",
            "description": "自定义API端点",
        },
    }

    @classmethod
    def get_platform_config(cls, platform: str) -> Dict[str, str]:
        """获取指定平台的配置"""
        if platform not in cls.PLATFORMS:
            raise ValueError(
                f"不支持的平台: {platform}。支持的平台: {list(cls.PLATFORMS.keys())}"
            )
        return cls.PLATFORMS[platform]

    @classmethod
    def list_platforms(cls) -> Dict[str, str]:
        """列出所有支持的平台（名称 -> 描述）"""
        return {name: cfg["description"] for name, cfg in cls.PLATFORMS.items()}


class Config:
    """配置管理类"""

    # 默认平台
    DEFAULT_PLATFORM = "siliconflow"

    # 从环境变量获取平台配置
    CURRENT_PLATFORM = os.getenv("LLM_PLATFORM", DEFAULT_PLATFORM)

    # 通用配置
    TEMPERATURE = 0.6
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    # 多线程配置
    MAX_WORKERS = 8          # 最大并发线程数（默认8个）
    QUEUE_TIMEOUT = 30       # 队列超时时间（秒）
    BATCH_SIZE = 10          # 批处理大小

    # 默认行为配置
    DEFAULT_SHOW_STATS = True        # 默认显示统计信息
    DEFAULT_CONCURRENT = True        # 默认使用多线程并发处理
    DEFAULT_SORT_OUTPUT = False      # 默认完成后按chunk_id排序输出
    DEFAULT_SAVE_CHUNK_TEXT = False  # 默认不保存原始chunk文本
    FAIL_ON_PARSE_ERROR = False      # 解析模型响应失败时是否抛异常（默认记录并跳过）

    # 回复关系标注配置
    REPLY_WINDOW = 6                 # 仅本 chunk 内向前看的窗口大小
    REPLY_CONFIDENCE_TH = 0.65       # 低于此阈值则视为不确定 -> reply=null

    @classmethod
    def _api_key_env_candidates(
        cls, platform: str, platform_config: Dict[str, Any]
    ) -> List[str]:
        """返回平台 API Key 环境变量候选（主变量 + 兼容别名）。"""
        candidates = [str(platform_config["api_key_env"])]
        # DeepSeek 兼容旧变量名，避免 launcher 历史配置失效。
        if platform == "deepseek":
            candidates.append("DEEPSEEK_API_KEY")
        # 去重并保持顺序
        seen = set()
        out: List[str] = []
        for name in candidates:
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @classmethod
    def _resolve_api_key(cls, platform: str, platform_config: Dict[str, Any]) -> str:
        for env_name in cls._api_key_env_candidates(platform, platform_config):
            value = os.getenv(env_name)
            if value:
                return value
        return ""

    @classmethod
    def get_current_platform_config(cls) -> Dict[str, Any]:
        """获取当前平台的配置"""
        platform_config = ModelPlatform.get_platform_config(cls.CURRENT_PLATFORM)

        api_key_envs = cls._api_key_env_candidates(cls.CURRENT_PLATFORM, platform_config)
        base_url_env = platform_config["base_url_env"]

        api_key = cls._resolve_api_key(cls.CURRENT_PLATFORM, platform_config)
        base_url = os.getenv(base_url_env, platform_config["default_base_url"])
        model_name = os.getenv(
            f"{cls.CURRENT_PLATFORM.upper()}_MODEL_NAME",
            platform_config["default_model"],
        )

        if not api_key:
            # 查找环境中已配置的其他平台（仅用于错误提示）
            available_platforms = [
                name
                for name, cfg in ModelPlatform.PLATFORMS.items()
                if cls._resolve_api_key(name, cfg)
            ]

            if available_platforms:
                raise ValueError(
                    f"平台 {cls.CURRENT_PLATFORM} 的API密钥未设置。\n"
                    f"请设置环境变量 {' 或 '.join(api_key_envs)}，\n"
                    f"或使用其他已配置的平台: {', '.join(available_platforms)}\n"
                    f"可通过设置 LLM_PLATFORM 环境变量切换平台"
                )
            else:
                env_list = [
                    f"  - {' 或 '.join(cls._api_key_env_candidates(name, cfg))}"
                    for name, cfg in ModelPlatform.PLATFORMS.items()
                ]
                raise ValueError(
                    f"未找到任何已配置的API密钥。\n"
                    f"请至少设置一个平台的API密钥:\n"
                    f"{chr(10).join(env_list)}"
                )

        return {
            "platform": cls.CURRENT_PLATFORM,
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "description": platform_config["description"],
        }

    @classmethod
    def set_platform(cls, platform: str) -> None:
        """设置当前平台"""
        if platform not in ModelPlatform.PLATFORMS:
            raise ValueError(f"不支持的平台: {platform}")
        cls.CURRENT_PLATFORM = platform

    # 文本处理配置
    MAX_TOKEN_LEN = 1000
    COVER_CONTENT = 100
    ENCODING = "cl100k_base"

    # 输出配置
    OUTPUT_ENCODING = "utf-8"
    OUTPUT_FORMAT = "jsonl"

    # Chunk-ID 配置
    INCLUDE_CHUNK_ID = True       # 是否在输出中包含chunk-id
    SAVE_CHUNK_TEXT = DEFAULT_SAVE_CHUNK_TEXT  # 与 DEFAULT_SAVE_CHUNK_TEXT 保持一致的实际开关
    BUFFER_BEFORE_WRITE = True    # 是否缓冲结果后按顺序写入

    # 默认对话提取模式
    DEFAULT_SCHEMA: Dict[str, Any] = {
        "task_description": "从小说中提取角色对话",
        "attributes": [
            {"name": "role", "description": "说话的角色名称", "type": "String"},
            {"name": "dialogue", "description": "角色说的对话内容", "type": "String"},
            {
                "name": "reply",
                "description": (
                    "必须包含 reply。若该发言明确在回复本 chunk 内前文某条，"
                    "则为对象；否则为 null。reply = "
                    "{target_index:int(>=0), target_role:string, confidence:float(0..1)}"
                ),
                "type": "Dict|null",
            },
        ],
        "example": [
            {
                "text": "……（示例文本）",
                "script": [
                    {"role": "A", "dialogue": "你在哪？", "reply": None},
                    {
                        "role": "B",
                        "dialogue": "在港口。",
                        "reply": {"target_index": 0, "target_role": "A", "confidence": 0.90},
                    },
                ],
            }
        ],
    }

    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "dialogue_chain.log"

    # 进度恢复配置
    CACHE_DIR = ".cache"
    PROGRESS_FILE = "progress.json"

    @classmethod
    def validate_config(cls) -> List[str]:
        """验证配置，返回错误列表"""
        errors: List[str] = []

        def is_int(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        def is_finite_number(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )

        try:
            cls.get_current_platform_config()
        except ValueError as e:
            errors.append(str(e))

        if not is_int(cls.MAX_TOKEN_LEN) or cls.MAX_TOKEN_LEN <= 0:
            errors.append("MAX_TOKEN_LEN必须是正整数")
        if not is_int(cls.COVER_CONTENT) or cls.COVER_CONTENT < 0:
            errors.append("COVER_CONTENT必须是非负整数")
        if (
            is_int(cls.MAX_TOKEN_LEN)
            and cls.MAX_TOKEN_LEN > 0
            and is_int(cls.COVER_CONTENT)
            and cls.COVER_CONTENT >= cls.MAX_TOKEN_LEN
        ):
            errors.append("MAX_TOKEN_LEN必须大于COVER_CONTENT")

        if not is_int(cls.MAX_WORKERS) or cls.MAX_WORKERS <= 0:
            errors.append("MAX_WORKERS必须是正整数")
        if not is_int(cls.MAX_RETRIES) or cls.MAX_RETRIES <= 0:
            errors.append("MAX_RETRIES必须是正整数")
        if not is_int(cls.REPLY_WINDOW) or cls.REPLY_WINDOW <= 0:
            errors.append("REPLY_WINDOW必须是正整数")

        if not is_finite_number(cls.TEMPERATURE) or not (0 <= float(cls.TEMPERATURE) <= 2):
            errors.append("TEMPERATURE必须是0-2之间的有限数值")
        if not is_finite_number(cls.RETRY_DELAY) or float(cls.RETRY_DELAY) < 0:
            errors.append("RETRY_DELAY必须是非负有限数值")
        if not is_finite_number(cls.REPLY_CONFIDENCE_TH) or not (
            0 <= float(cls.REPLY_CONFIDENCE_TH) <= 1
        ):
            errors.append("REPLY_CONFIDENCE_TH必须是0-1之间的有限数值")

        return errors

    @classmethod
    def get_system_prompt_template(cls) -> str:
        """获取系统提示模板"""
        return """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.
{TypeScript}
Please output the extracted information in JSON format. 

Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: 
{Input}
Output:
{Output}

"""

    @classmethod
    def get_typescript_template(cls) -> str:
        """获取TypeScript模板"""
        return """
```TypeScript
    script: Array<Dict( // {task_description}
    {attributes}
    )>
```
"""
