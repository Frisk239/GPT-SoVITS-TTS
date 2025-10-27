"""
语音服务API路由
提供语音合成功能的REST API
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io

from app.services.deepseek_service import DeepSeekService
from app.services.gpt_sovits_service import GPTSoVITSService

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    page: Optional[str] = "tts-chat"

class SynthesisRequest(BaseModel):
    text: str
    page: Optional[str] = "tts-chat"

@router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """
    与AI对话接口

    Args:
        request: 包含用户消息和页面标识的请求

    Returns:
        AI回复内容
    """
    try:
        logger.info(f"💬 收到对话请求: {request.message[:50]}...")

        # 获取页面配置
        page_config = gpt_sovits_service.get_page_config(request.page)
        personality = page_config.get("personality", "")
        chat_config = page_config.get("chat_config", {})

        # 调用DeepSeek生成回复
        ai_response = await deepseek_service.generate_fujian_response(
            user_message=request.message,
            personality=personality
        )

        logger.info(f"✅ AI回复生成完成: {len(ai_response)} 字符")

        return {
            "success": True,
            "response": ai_response,
            "page": request.page
        }

    except Exception as e:
        logger.error(f"❌ 对话请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"对话服务异常: {str(e)}")

@router.post("/synthesize")
async def synthesize_speech(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    语音合成接口

    Args:
        request: 包含文本和页面标识的请求

    Returns:
        音频流
    """
    try:
        # 编码验证和日志
        logger.info(f"🎵 收到语音合成请求 - 原始文本: {repr(request.text)}")
        logger.info(f"🎵 收到语音合成请求 - 显示文本: {request.text[:50]}...")

        # 验证文本编码
        if not request.text or request.text.strip() == "":
            raise HTTPException(status_code=400, detail="文本不能为空")

        # 检查是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in request.text)
        logger.info(f"🎵 文本包含中文字符: {has_chinese}")

        # 如果文本是乱码，尝试提示用户
        if '??' in request.text and has_chinese:
            logger.warning("⚠️ 检测到可能的编码问题，请确保客户端使用UTF-8编码")

        # 调用语音合成服务
        audio_data = await gpt_sovits_service.synthesize_speech(
            text=request.text,
            page=request.page
        )

        if not audio_data:
            raise HTTPException(status_code=500, detail="语音合成失败")

        # 返回音频流
        audio_stream = io.BytesIO(audio_data)

        logger.info(f"✅ 语音合成完成: {len(audio_data)} bytes")

        return StreamingResponse(
            audio_stream,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )

    except Exception as e:
        logger.error(f"❌ 语音合成请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音合成服务异常: {str(e)}")

@router.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 检查各个服务状态
        deepseek_status = await deepseek_service.health_check()
        gpt_sovits_status = await gpt_sovits_service.health_check()

        overall_status = "healthy"
        if deepseek_status.get("status") != "healthy" or gpt_sovits_status.get("service") != "gpt_sovits":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "services": {
                "deepseek": deepseek_status,
                "gpt_sovits": gpt_sovits_status
            },
            "timestamp": deepseek_status.get("last_check", "")
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/config/{page}")
async def get_page_config(page: str):
    """获取页面配置"""
    try:
        config = gpt_sovits_service.get_page_config(page)
        if not config:
            raise HTTPException(status_code=404, detail=f"页面配置不存在: {page}")

        return {
            "page": page,
            "config": config
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取页面配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置异常: {str(e)}")

# 全局服务实例
deepseek_service = None
gpt_sovits_service = None

def init_services():
    """初始化服务实例"""
    global deepseek_service, gpt_sovits_service

    import os
    from dotenv import load_dotenv

    # 加载环境变量
    load_dotenv()

    # 初始化DeepSeek服务
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        deepseek_service = DeepSeekService(api_key)
    else:
        logger.warning("未设置DEEPSEEK_API_KEY")

    # 初始化GPT-SoVITS服务
    gpt_sovits_service = GPTSoVITSService()

# 在模块导入时初始化服务
init_services()
