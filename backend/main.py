"""
GPT-SoVITS语音合成后端服务
基于FastAPI提供语音合成功能
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="GPT-SoVITS语音合成API",
    description="为闽仔AI助手提供语音合成功能的后端服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入路由
from app.routes.voice_service import router as voice_router

# 注册路由
app.include_router(
    voice_router,
    prefix="/api/voice",
    tags=["语音服务"]
)

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 GPT-SoVITS后端服务启动中...")

    # 检查环境变量
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        logger.warning("⚠️ 未设置DEEPSEEK_API_KEY环境变量")

    # 检查配置文件
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        logger.warning(f"⚠️ 配置文件不存在: {config_path}")

    logger.info("✅ GPT-SoVITS后端服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🛑 GPT-SoVITS后端服务关闭")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "GPT-SoVITS语音合成API服务",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "gpt-sovits-backend"
    }

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    logger.info(f"启动服务器: {host}:{port}, debug={debug}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
