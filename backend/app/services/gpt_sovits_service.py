"""
GPT-SoVITS语音合成服务
基于GPT-SoVITS-v2pro源码完全集成
专门为闽仔角色提供真实语音合成功能
"""

import json
import logging
import os
import sys
import asyncio
import gc
import math
import random
import time
import traceback
from copy import deepcopy
from typing import Dict, List, Optional, Any, Tuple, Union, Generator

import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from tqdm import tqdm
import ffmpeg
import librosa
import soundfile as sf
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer

# GPT_SoVITS 动态导入模块
# 不使用直接导入，改为运行时动态导入

logger = logging.getLogger(__name__)

# 音频重采样缓存
resample_transform_dict = {}

def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

# 语言设置
language = os.environ.get("language", "Auto")
# 简化语言设置，避免在模块级别使用未导入的函数
i18n = None  # 将在需要时动态初始化

# 频谱归一化参数
spec_min = -12
spec_max = 2

def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1

def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min

# 梅尔频谱函数
mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    """变速处理音频"""
    raw_audio = input_audio.astype(np.int16).tobytes()
    input_stream = ffmpeg.input("pipe:", format="s16le", acodec="pcm_s16le", ar=str(sr), ac=1)
    output_stream = input_stream.filter("atempo", speed)
    out, _ = output_stream.output("pipe:", format="s16le", acodec="pcm_s16le").run(
        input=raw_audio, capture_stdout=True, capture_stderr=True
    )
    processed_audio = np.frombuffer(out, np.int16)
    return processed_audio

def set_seed(seed: int):
    """设置随机种子"""
    seed = int(seed)
    seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
    print(f"Set seed to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed

class DictToAttrRecursive(dict):
    """字典转属性递归类"""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class NO_PROMPT_ERROR(Exception):
    pass

class GPTSoVITSService:
    """GPT-SoVITS推理服务"""

    def __init__(self, config_path: str = "./config.json"):
        # 计算绝对路径
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "../../..")
            self.config_path = os.path.join(project_root, "config.json")
        else:
            self.config_path = config_path

        self.config = self._load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 计算GPT_SoVITS路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
        self.gpt_sovits_path = os.path.join(self.project_root, "backend", "GPT_SoVITS")

        # GPT-SoVITS TTS实例
        self.tts_pipeline = None

        # 模型路径（使用绝对路径）
        self.gpt_weights_dir = os.path.join(self.project_root, "models", "GPT-SoVITS", "GPT_weights_v2Pro")
        self.sovits_weights_dir = os.path.join(self.project_root, "models", "GPT-SoVITS", "SoVITS_weights_v2Pro")

        # 模型缓存
        self.models_cache = {}

        # 动态导入的模块缓存
        self._modules_cache = {}

        # 设置模块路径
        self._setup_module_paths()

        # 初始化TTS管道
        self._init_tts_pipeline()

    def _setup_module_paths(self):
        """设置GPT-SoVITS模块路径到sys.path"""
        try:
            # 下载NLTK数据包
            logger.info("📥 下载NLTK数据包...")
            import nltk
            try:
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                logger.info("✅ NLTK数据包下载完成")
            except Exception as e:
                logger.warning(f"⚠️ NLTK数据包下载失败: {e}")

            paths_to_add = [
                self.gpt_sovits_path,  # 根目录
                os.path.join(self.gpt_sovits_path, "AR"),
                os.path.join(self.gpt_sovits_path, "AR", "models"),
                os.path.join(self.gpt_sovits_path, "AR", "modules"),
                os.path.join(self.gpt_sovits_path, "BigVGAN"),
                os.path.join(self.gpt_sovits_path, "module"),
                os.path.join(self.gpt_sovits_path, "tools"),
                os.path.join(self.gpt_sovits_path, "tools", "i18n"),
                os.path.join(self.gpt_sovits_path, "TTS_infer_pack"),
                os.path.join(self.gpt_sovits_path, "feature_extractor"),
                os.path.join(self.gpt_sovits_path, "text"),
            ]

            for path in paths_to_add:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
                    logger.info(f"✅ 添加GPT-SoVITS路径: {path}")

            logger.info(f"📂 GPT_SoVITS sys.path设置完成，总共添加 {len(paths_to_add)} 个路径")

        except Exception as e:
            logger.error(f"❌ 设置模块路径失败: {e}")

    def _import_module_from_file(self, relative_path: str, class_name: str = None):
        """从文件动态导入模块或类"""
        try:
            import importlib.util

            module_path = os.path.join(self.gpt_sovits_path, relative_path)
            if not os.path.exists(module_path):
                logger.error(f"模块文件不存在: {module_path}")
                return None

            # 创建模块名（基于相对路径）
            module_name = relative_path.replace("/", ".").replace("\\", ".").replace(".py", "")

            # 检查缓存
            if module_name in self._modules_cache:
                module = self._modules_cache[module_name]
            else:
                # 对于 TTS.py，先预导入其依赖的模块
                if "TTS.py" in relative_path:
                    self._preload_tts_dependencies()

                # 动态导入
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    logger.error(f"无法创建模块规格: {module_path}")
                    return None

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._modules_cache[module_name] = module
                logger.info(f"✅ 动态导入模块: {module_name}")

            # 如果指定了类名，返回类；否则返回模块
            if class_name:
                if hasattr(module, class_name):
                    return getattr(module, class_name)
                else:
                    logger.error(f"模块 {module_name} 中没有找到类 {class_name}")
                    return None

            return module

        except Exception as e:
            logger.error(f"❌ 动态导入失败 {relative_path}: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return None

    def _preload_tts_dependencies(self):
        """预加载TTS模块的依赖"""
        try:
            logger.info("🎯 预加载TTS依赖模块...")

            # 首先创建并注册GPT_SoVITS包
            self._register_gpt_sovits_package()

            # 预导入关键模块
            dependencies = [
                "AR/models/t2s_lightning_module.py",
                "BigVGAN/bigvgan.py",
                "feature_extractor/cnhubert.py",
                "module/mel_processing.py",
                "module/models.py",
                "process_ckpt.py",
                "tools/audio_sr.py",
                "tools/i18n/i18n.py",
                "TTS_infer_pack/text_segmentation_method.py",
                "TTS_infer_pack/TextPreprocessor.py",
                "sv.py"
            ]

            for dep in dependencies:
                try:
                    self._import_module_from_file(dep)
                except Exception as e:
                    logger.warning(f"预加载依赖失败 {dep}: {e}")
                    continue

            logger.info("✅ TTS依赖预加载完成")

        except Exception as e:
            logger.error(f"❌ 预加载TTS依赖失败: {e}")

    def _register_gpt_sovits_package(self):
        """注册GPT_SoVITS包到sys.modules"""
        try:
            import types
            import sys

            # 创建GPT_SoVITS包对象
            gpt_sovits_package = types.ModuleType('GPT_SoVITS')
            gpt_sovits_package.__path__ = [self.gpt_sovits_path]
            gpt_sovits_package.__file__ = os.path.join(self.gpt_sovits_path, '__init__.py')

            # 注册到sys.modules
            sys.modules['GPT_SoVITS'] = gpt_sovits_package

            # 递归创建子包
            self._create_subpackages(gpt_sovits_package, self.gpt_sovits_path)

            logger.info("✅ GPT_SoVITS包注册完成")

        except Exception as e:
            logger.error(f"❌ 注册GPT_SoVITS包失败: {e}")

    def _create_subpackages(self, parent_package, parent_path):
        """递归创建子包"""
        try:
            import types

            # 遍历子目录
            for item in os.listdir(parent_path):
                item_path = os.path.join(parent_path, item)
                if os.path.isdir(item_path):
                    # 检查是否有__init__.py
                    init_file = os.path.join(item_path, '__init__.py')
                    if os.path.exists(init_file) or item in ['f5_tts', 'AR', 'BigVGAN', 'module', 'tools', 'TTS_infer_pack', 'feature_extractor', 'text']:
                        # 创建子包
                        subpackage_name = f"{parent_package.__name__}.{item}"
                        subpackage = types.ModuleType(subpackage_name)
                        subpackage.__path__ = [item_path]
                        subpackage.__file__ = init_file if os.path.exists(init_file) else item_path

                        # 设置父包引用
                        setattr(parent_package, item, subpackage)
                        sys.modules[subpackage_name] = subpackage

                        # 递归创建子包的子包
                        self._create_subpackages(subpackage, item_path)

        except Exception as e:
            logger.warning(f"创建子包失败 {parent_path}: {e}")

    def _init_tts_pipeline(self):
        """初始化TTS管道"""
        try:
            logger.info("🎵 初始化GPT-SoVITS TTS管道...")
            # TTS管道将在第一次推理时动态创建
            self.tts_pipeline = None
            logger.info("✅ TTS管道初始化完成（延迟加载）")
        except Exception as e:
            logger.error(f"❌ TTS管道初始化失败: {e}")
            self.tts_pipeline = None

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            # 尝试多个可能路径
            possible_paths = [
                self.config_path,  # 相对路径
                os.path.join(os.path.dirname(__file__), "../..", self.config_path),  # 向上两级
                os.path.join(os.path.dirname(__file__), "../../../", self.config_path)  # 项目根目录
            ]

            logger.info(f"🔍 尝试加载配置文件，尝试路径: {possible_paths}")

            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            logger.info(f"✅ 成功从 {path} 加载配置文件")
                            logger.info(f"📋 配置内容: pages={list(config.get('pages', {}).keys())}, default_page={config.get('default_page')}")
                            return config
                    except Exception as e:
                        logger.error(f"❌ 加载配置文件失败 {path}: {e}")
                        continue
                else:
                    logger.debug(f"⚠️ 配置文件不存在: {path}")

            logger.warning(f"⚠️ 所有配置文件路径都不存在，使用默认配置")
            return self._get_default_config()

        except Exception as e:
            logger.error(f"❌ 加载配置文件异常: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置 - 从config.json文件读取"""
        try:
            # 尝试读取config.json文件
            config_path = os.path.join(self.project_root, "backend", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info("✅ 从config.json成功加载配置")
                    return config
            else:
                logger.warning(f"⚠️ config.json文件不存在: {config_path}")
        except Exception as e:
            logger.error(f"❌ 读取config.json失败: {e}")

        # 如果读取失败，返回基本的默认配置
        logger.warning("⚠️ 使用基本默认配置")
        return {
            "pages": {},
            "default_page": "tts-chat",
            "synthesis_params": {
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "speed": 1.2,
                "noise_scale": 0.5,
                "text_split_method": "cut5",
                "batch_size": 1,
                "fragment_interval": 0.3,
                "seed": -1,
                "parallel_infer": True,
                "repetition_penalty": 1.35
            },
            "model_paths": {
                "gpt_weights_dir": "../models/GPT-SoVITS/GPT_weights_v2Pro",
                "sovits_weights_dir": "../models/GPT-SoVITS/SoVITS_weights_v2Pro"
            }
        }

    async def synthesize_speech(
        self,
        text: str,
        page: str = "tts-chat"
    ) -> bytes:
        """
        语音合成

        Args:
            text: 要合成的文本
            page: 页面标识，用于获取对应配置

        Returns:
            音频字节数据
        """
        try:
            # 获取页面配置
            page_config = self.config.get("pages", {}).get(page, {})
            voice_config = page_config.get("voice_config", {})

            if not voice_config:
                logger.error(f"❌ 页面 '{page}' 的语音配置不存在")
                return b""

            # 获取模型路径
            gpt_model = voice_config.get("gpt_model")
            sovits_model = voice_config.get("sovits_model")

            if not gpt_model or not sovits_model:
                logger.error(f"❌ 页面 '{page}' 的模型配置不完整")
                return b""

            # 检查模型文件是否存在
            gpt_path = os.path.join(self.gpt_weights_dir, gpt_model)
            sovits_path = os.path.join(self.sovits_weights_dir, sovits_model)

            if not os.path.exists(gpt_path) or not os.path.exists(sovits_path):
                logger.error(f"❌ 模型文件不存在: GPT={gpt_path}, SoVITS={sovits_path}")
                return b""

            logger.info(f"🎵 开始合成语音: '{text}' (页面: {page})")

            # 调用真实的GPT-SoVITS推理
            audio_data = await self._run_inference(
                text, gpt_path, sovits_path, voice_config.get("voice_params", {})
            )

            logger.info(f"✅ 语音合成完成，音频大小: {len(audio_data)} bytes")
            return audio_data

        except Exception as e:
            logger.error(f"❌ 语音合成失败: {e}")
            return b""

    async def _run_inference(
        self,
        text: str,
        gpt_path: str,
        sovits_path: str,
        voice_params: Dict
    ) -> bytes:
        """
        执行GPT-SoVITS推理

        基于GPT-SoVITS源码的完整推理流程
        """
        try:
            logger.info("🎯 开始GPT-SoVITS推理流程...")

            # 1. 创建TTS配置
            tts_config = self._create_tts_config(gpt_path, sovits_path)
            logger.info("✅ TTS配置创建完成")

            # 2. 初始化TTS管道
            TTS_class = self._import_module_from_file("TTS_infer_pack/TTS.py", "TTS")
            if TTS_class is None:
                logger.error("❌ 无法导入TTS类")
                return b""
            tts_pipeline = TTS_class(tts_config)
            logger.info("✅ TTS管道初始化完成")

            # 3. 获取角色配置
            role_config = self._get_role_config_by_model(gpt_path, sovits_path)
            if not role_config:
                logger.error("❌ 未找到角色配置")
                return b""

            # 4. 获取参考音频路径
            ref_audio_path = role_config.get("ref_audio_path")
            if not ref_audio_path or not os.path.exists(ref_audio_path):
                logger.error(f"❌ 参考音频不存在: {ref_audio_path}")
                return b""

            # 5. 设置参考音频
            tts_pipeline.set_ref_audio(ref_audio_path)
            logger.info(f"✅ 参考音频设置完成: {ref_audio_path}")

            # 6. 准备推理参数
            inference_params = {
                "text": text,
                "text_lang": "zh",  # 中文
                "ref_audio_path": ref_audio_path,
                "prompt_text": role_config.get("prompt_text", ""),
                "prompt_lang": "zh",
                "top_k": 5,
                "top_p": 1.0,
                "temperature": 1.0,
                "text_split_method": "cut5",
                "batch_size": 1,
                "speed_factor": voice_params.get("speed", 1.0),
                "fragment_interval": 0.3,
                "seed": -1,
                "parallel_infer": False,  # 禁用并行处理，保证句子顺序
                "repetition_penalty": 1.35
            }

            logger.info(f"🎵 开始语音合成: '{text}'")

            # 7. 执行推理
            sr, audio_data = next(tts_pipeline.run(inference_params))

            # 8. 转换为16bit PCM
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32768).astype(np.int16)

            # 9. 创建WAV文件
            wav_data = self._create_wav_file(audio_data.tobytes(), sr)

            logger.info(f"✅ 推理完成，音频大小: {len(wav_data)} bytes, 采样率: {sr}Hz")

            # 10. 清理资源
            del tts_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return wav_data

        except Exception as e:
            logger.error(f"❌ GPT-SoVITS推理失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return b""

    def _create_tts_config(self, gpt_path: str, sovits_path: str):
        """创建TTS配置字典"""
        # 计算预训练模型的绝对路径
        pretrained_dir = os.path.join(self.gpt_sovits_path, "pretrained_models")
        bert_path = os.path.join(pretrained_dir, "chinese-roberta-wwm-ext-large")
        cnhubert_path = os.path.join(pretrained_dir, "chinese-hubert-base")

        # 创建custom配置（TTS_Config期望的格式）
        custom_config = {
            "device": self.device,
            "is_half": True if self.device == "cuda" else False,
            "version": "v2Pro",
            "t2s_weights_path": gpt_path,
            "vits_weights_path": sovits_path,
            "bert_base_path": bert_path,
            "cnhuhbert_base_path": cnhubert_path
        }

        # 返回包含custom键的配置字典
        return {"custom": custom_config}

    def _get_role_config_by_model(self, gpt_path: str, sovits_path: str) -> Optional[Dict]:
        """根据模型路径获取角色配置"""
        gpt_model = os.path.basename(gpt_path)
        sovits_model = os.path.basename(sovits_path)

        for page_name, page_config in self.config.get("pages", {}).items():
            voice_config = page_config.get("voice_config", {})
            if (voice_config.get("gpt_model") == gpt_model and
                voice_config.get("sovits_model") == sovits_model):
                # 添加参考音频路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                ref_audio_name = voice_config.get("ref_audio_path", "")
                # 尝试多个可能的路径
                possible_paths = [
                    os.path.join(current_dir, "../../../models/GPT-SoVITS/GPT-SoVITS-slice", ref_audio_name.split('/')[-1]),
                    os.path.join(current_dir, "../../../models/GPT-SoVITS/GPT-SoVITS-slice", ref_audio_name.split('/')[-1].replace("-slicer", "")),
                    os.path.join(current_dir, "../../../models/GPT-SoVITS/GPT-SoVITS-slice", f"{page_config.get('role', 'minpaixinyu')}.wav"),
                    os.path.join(current_dir, "../../../models/GPT-SoVITS/GPT-SoVITS-slice", f"{gpt_model.split('-')[0]}.wav")
                ]

                ref_audio_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        ref_audio_path = path
                        break

                if not ref_audio_path:
                    logger.warning(f"⚠️ 参考音频不存在: {possible_paths[0]}")
                    # 使用第一个可能的路径作为默认值
                    ref_audio_path = possible_paths[0]

                return {
                    **voice_config,
                    "ref_audio_path": ref_audio_path,
                    "prompt_text": voice_config.get("ref_audio_text", "")
                }

        return None

    def _create_wav_file(self, pcm_data: bytes, sample_rate: int = 44100) -> bytes:
        """
        创建WAV文件格式

        Args:
            pcm_data: PCM音频数据（16bit, 单声道）
            sample_rate: 采样率

        Returns:
            完整的WAV文件数据
        """
        try:
            # WAV文件头结构
            # RIFF头
            riff_header = b'RIFF'
            file_size = 36 + len(pcm_data)  # 36是WAV头的固定大小
            riff_size = file_size.to_bytes(4, 'little')

            # WAVE标识
            wave_header = b'WAVE'

            # fmt子块
            fmt_header = b'fmt '
            fmt_size = (16).to_bytes(4, 'little')  # fmt子块大小
            audio_format = (1).to_bytes(2, 'little')  # PCM格式
            num_channels = (1).to_bytes(2, 'little')  # 单声道
            sample_rate_bytes = sample_rate.to_bytes(4, 'little')
            byte_rate = (sample_rate * 1 * 16 // 8).to_bytes(4, 'little')  # 字节率
            block_align = (1 * 16 // 8).to_bytes(2, 'little')  # 块对齐
            bits_per_sample = (16).to_bytes(2, 'little')  # 16位

            # data子块
            data_header = b'data'
            data_size = len(pcm_data).to_bytes(4, 'little')

            # 组合所有部分
            wav_file = (
                riff_header + riff_size + wave_header +
                fmt_header + fmt_size + audio_format + num_channels +
                sample_rate_bytes + byte_rate + block_align + bits_per_sample +
                data_header + data_size + pcm_data
            )

            logger.info(f"✅ WAV文件创建成功: {len(wav_file)} bytes, 采样率: {sample_rate}Hz")
            return wav_file

        except Exception as e:
            logger.error(f"❌ 创建WAV文件失败: {e}")
            return b""

    def get_page_config(self, page: str) -> Dict:
        """获取页面配置"""
        return self.config.get("pages", {}).get(page, {})

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            page_config = self.get_page_config("tts-chat")
            voice_config = page_config.get("voice_config", {})

            gpt_model = voice_config.get("gpt_model")
            sovits_model = voice_config.get("sovits_model")

            gpt_exists = os.path.exists(os.path.join(self.gpt_weights_dir, gpt_model or ""))
            sovits_exists = os.path.exists(os.path.join(self.sovits_weights_dir, sovits_model or ""))

            return {
                "service": "gpt_sovits",
                "device": self.device,
                "gpt_model_exists": gpt_exists,
                "sovits_model_exists": sovits_exists,
                "gpt_weights_dir": self.gpt_weights_dir,
                "sovits_weights_dir": self.sovits_weights_dir,
                "config_loaded": bool(self.config)
            }

        except Exception as e:
            return {
                "service": "gpt_sovits",
                "error": str(e),
                "device": self.device
            }

# 全局服务实例将在main.py中初始化
