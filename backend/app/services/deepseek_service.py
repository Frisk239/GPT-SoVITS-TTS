"""
DeepSeek AI服务
处理与DeepSeek API的对话交互
"""

import json
import logging
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DeepSeekService:
    """DeepSeek AI对话服务"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-chat",
        temperature: float = 0.8,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用DeepSeek API进行对话

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            API响应结果
        """
        try:
            url = f"{self.base_url}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            logger.info(f"🤖 发送DeepSeek请求: {len(messages)} 条消息")

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]

                        logger.info(f"✅ DeepSeek响应成功: {len(content)} 字符")
                        return {
                            "success": True,
                            "response": content,
                            "usage": result.get("usage", {}),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ DeepSeek API错误: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "error": f"API请求失败: {response.status}",
                            "timestamp": datetime.now().isoformat()
                        }

        except Exception as e:
            logger.error(f"❌ DeepSeek服务异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def generate_fujian_response(
        self,
        user_message: str,
        context: Optional[List[Dict]] = None,
        personality: str = ""
    ) -> str:
        """
        生成福建文化相关的回复

        Args:
            user_message: 用户消息
            context: 对话上下文
            personality: 角色人设

        Returns:
            AI回复内容
        """
        try:
            system_prompt = f"""你是一个名为闽仔的AI助手，专门介绍福建文化和历史。

{personality}

请用温柔亲切、知识丰富的语气回答用户的问题。你的回答应该：
1. 准确介绍福建的历史、文化和风景
2. 用生动的语言描述，让用户感受到福建文化的魅力
3. 保持积极友好的态度
4. 如果用户问的问题与福建无关，可以礼貌地引导到福建文化话题

记住：你是闽仔，不是其他AI助手。"""

            messages = [{"role": "system", "content": system_prompt}]

            # 添加上下文
            if context:
                for msg in context[-5:]:  # 只保留最近5条消息作为上下文
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # 添加当前用户消息
            messages.append({"role": "user", "content": user_message})

            result = await self.chat_completion(
                messages=messages,
                temperature=0.8,
                max_tokens=800
            )

            if result["success"]:
                return result["response"]
            else:
                logger.error(f"生成回复失败: {result.get('error', '未知错误')}")
                return "抱歉，我现在有点小问题，请稍后再试试吧！😅"

        except Exception as e:
            logger.error(f"生成福建文化回复异常: {e}")
            return "抱歉，我现在有点小问题，请稍后再试试吧！😅"

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 发送一个简单的测试请求
            test_messages = [
                {"role": "user", "content": "你好"}
            ]

            result = await self.chat_completion(
                messages=test_messages,
                max_tokens=10
            )

            return {
                "service": "deepseek",
                "status": "healthy" if result["success"] else "unhealthy",
                "api_key_configured": bool(self.api_key),
                "base_url": self.base_url,
                "last_check": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "service": "deepseek",
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

# 全局服务实例将在main.py中初始化
