"""
OpenAudit Model Manager
Comprehensive AI model management with Google-level abstraction
Supports OpenAI, Anthropic, xAI (Grok), and extensible provider architecture
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

# Core LangChain imports
from langchain_core.messages import HumanMessage

# Provider-specific imports
from langchain_openai import ChatOpenAI

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from langchain_anthropic import ChatAnthropic

    ANTHROPIC_AVAILABLE = True
    logger.info("Anthropic provider available")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available - install langchain-anthropic for Claude support")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GOOGLE_AVAILABLE = True
    logger.info("Google Gemini provider available")
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google Gemini not available - install langchain-google-genai for support")


class ModelProvider(Enum):
    """AI model providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"  # Grok
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for an AI model"""

    name: str
    provider: ModelProvider
    model_id: str
    display_name: str
    category: str  # reasoning, general, fast, etc.
    max_tokens: Optional[int] = None
    supports_temperature: bool = True
    supports_streaming: bool = True
    cost_per_1k_tokens: Optional[float] = None
    description: str = ""


@dataclass
class ModelResponse:
    """Standardized response from any model"""

    model_name: str
    provider: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    latency_ms: int
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None


class ModelManager:
    """
    Comprehensive model management with Google-level engineering
    - Supports multiple providers
    - Extensible architecture
    - Proper error handling
    - Cost tracking
    - Performance monitoring
    """

    def __init__(self):
        self.models: Dict[str, BaseChatModel] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all available models from all providers"""
        print("🚀 Initializing comprehensive AI model suite...")

        # OpenAI Models
        self._setup_openai_models()

        # Anthropic Models (Claude)
        if ANTHROPIC_AVAILABLE:
            self._setup_anthropic_models()

        # xAI Models (Grok)
        self._setup_xai_models()

        # Google Models (Gemini)
        if GOOGLE_AVAILABLE:
            self._setup_google_models()

        # Custom/Open Source Models
        self._setup_custom_models()

        print(
            f"✅ Initialized {len(self.models)} models across {len(set(c.provider for c in self.model_configs.values()))} providers"
        )
        self._print_model_summary()

    def _setup_openai_models(self):
        """Setup comprehensive OpenAI model suite"""
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  No OpenAI API key found")
            return

        print("🔵 Setting up OpenAI models...")

        # Define OpenAI model configurations
        openai_models = [
            # GPT-3.5 Series
            ModelConfig(
                "gpt-3.5-turbo",
                ModelProvider.OPENAI,
                "gpt-3.5-turbo",
                "GPT-3.5 Turbo",
                "fast",
                cost_per_1k_tokens=0.002,
                description="Fast, cost-effective model for most tasks",
            ),
            # GPT-4 Series
            ModelConfig(
                "gpt-4",
                ModelProvider.OPENAI,
                "gpt-4",
                "GPT-4",
                "general",
                cost_per_1k_tokens=0.06,
                description="Most capable GPT-4 model",
            ),
            ModelConfig(
                "gpt-4-turbo",
                ModelProvider.OPENAI,
                "gpt-4-turbo",
                "GPT-4 Turbo",
                "general",
                cost_per_1k_tokens=0.03,
                description="Faster GPT-4 with updated knowledge",
            ),
            ModelConfig(
                "gpt-4o",
                ModelProvider.OPENAI,
                "gpt-4o",
                "GPT-4o",
                "general",
                cost_per_1k_tokens=0.015,
                description="Optimized GPT-4 for speed and cost",
            ),
            ModelConfig(
                "gpt-4o-mini",
                ModelProvider.OPENAI,
                "gpt-4o-mini",
                "GPT-4o Mini",
                "fast",
                cost_per_1k_tokens=0.0075,
                description="Compact GPT-4o for high-volume tasks",
            ),
            # GPT-4.1 Series (if available)
            ModelConfig(
                "gpt-4.1-nano",
                ModelProvider.OPENAI,
                "gpt-4.1-nano",
                "GPT-4.1 Nano",
                "fast",
                cost_per_1k_tokens=0.005,
                description="Ultra-fast GPT-4.1 variant",
            ),
            ModelConfig(
                "gpt-4.1-mini",
                ModelProvider.OPENAI,
                "gpt-4.1-mini",
                "GPT-4.1 Mini",
                "general",
                cost_per_1k_tokens=0.01,
                description="Balanced GPT-4.1 model",
            ),
            ModelConfig(
                "gpt-4.1",
                ModelProvider.OPENAI,
                "gpt-4.1",
                "GPT-4.1",
                "general",
                cost_per_1k_tokens=0.03,
                description="Latest GPT-4.1 full model",
            ),
            # Reasoning Models (o1 Series)
            ModelConfig(
                "o1-preview",
                ModelProvider.OPENAI,
                "o1-preview",
                "o1 Preview",
                "reasoning",
                supports_temperature=False,
                cost_per_1k_tokens=0.15,
                description="Advanced reasoning model - preview version",
            ),
            ModelConfig(
                "o1-mini",
                ModelProvider.OPENAI,
                "o1-mini",
                "o1 Mini",
                "reasoning",
                supports_temperature=False,
                cost_per_1k_tokens=0.12,
                description="Compact reasoning model",
            ),
            ModelConfig(
                "o1",
                ModelProvider.OPENAI,
                "o1",
                "o1",
                "reasoning",
                supports_temperature=False,
                cost_per_1k_tokens=0.20,
                description="Full o1 reasoning model",
            ),
            # o3 Series (newest reasoning)
            ModelConfig(
                "o3-mini",
                ModelProvider.OPENAI,
                "o3-mini",
                "o3 Mini",
                "reasoning",
                supports_temperature=False,
                cost_per_1k_tokens=0.15,
                description="Latest compact reasoning model",
            ),
            ModelConfig(
                "o3",
                ModelProvider.OPENAI,
                "o3",
                "o3",
                "reasoning",
                supports_temperature=False,
                cost_per_1k_tokens=0.25,
                description="Most advanced reasoning model",
            ),
        ]

        for config in openai_models:
            try:
                model_kwargs = {
                    "model": config.model_id,
                    "temperature": 0,
                }

                # Add seed for reproducibility (not supported by reasoning models)
                if config.supports_temperature:
                    model_kwargs["seed"] = 42

                model = ChatOpenAI(**model_kwargs)
                self.models[config.name] = model
                self.model_configs[config.name] = config
                print(f"  ✓ {config.display_name}")

            except Exception as e:
                print(f"  ✗ {config.display_name}: {e}")

    def _setup_anthropic_models(self):
        """Setup Anthropic Claude models"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("⚠️  No Anthropic API key found")
            return

        print("🟠 Setting up Anthropic Claude models...")

        claude_models = [
            ModelConfig(
                "claude-3-5-sonnet",
                ModelProvider.ANTHROPIC,
                "claude-3-5-sonnet-20241022",
                "Claude 3.5 Sonnet",
                "general",
                cost_per_1k_tokens=0.015,
                description="Most capable Claude model",
            ),
            ModelConfig(
                "claude-3-5-haiku",
                ModelProvider.ANTHROPIC,
                "claude-3-5-haiku-20241022",
                "Claude 3.5 Haiku",
                "fast",
                cost_per_1k_tokens=0.008,
                description="Fast and efficient Claude model",
            ),
            ModelConfig(
                "claude-3-opus",
                ModelProvider.ANTHROPIC,
                "claude-3-opus-20240229",
                "Claude 3 Opus",
                "general",
                cost_per_1k_tokens=0.075,
                description="Most powerful Claude 3 model",
            ),
            ModelConfig(
                "claude-3-sonnet",
                ModelProvider.ANTHROPIC,
                "claude-3-sonnet-20240229",
                "Claude 3 Sonnet",
                "general",
                cost_per_1k_tokens=0.015,
                description="Balanced Claude 3 model",
            ),
            ModelConfig(
                "claude-3-haiku",
                ModelProvider.ANTHROPIC,
                "claude-3-haiku-20240307",
                "Claude 3 Haiku",
                "fast",
                cost_per_1k_tokens=0.0025,
                description="Fastest Claude 3 model",
            ),
        ]

        for config in claude_models:
            try:
                model = ChatAnthropic(
                    model=config.model_id, temperature=0, max_tokens=4096
                )
                self.models[config.name] = model
                self.model_configs[config.name] = config
                print(f"  ✓ {config.display_name}")

            except Exception as e:
                print(f"  ✗ {config.display_name}: {e}")

    def _setup_xai_models(self):
        """Setup xAI Grok models"""
        # Check for both XAI_API_KEY and GROK_API_KEY for backwards compatibility
        xai_api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not xai_api_key:
            print("⚠️  No xAI API key found (for Grok models)")
            print("    Set either XAI_API_KEY or GROK_API_KEY in your .env file")
            return

        print("🟣 Setting up xAI Grok models...")

        grok_models = [
            ModelConfig(
                "grok-beta",
                ModelProvider.XAI,
                "grok-beta",
                "Grok Beta",
                "general",
                cost_per_1k_tokens=0.05,
                description="xAI's Grok model - witty and capable",
            ),
            ModelConfig(
                "grok-vision-beta",
                ModelProvider.XAI,
                "grok-vision-beta",
                "Grok Vision Beta",
                "general",
                cost_per_1k_tokens=0.05,
                description="Grok with vision capabilities",
            ),
        ]

        for config in grok_models:
            try:
                # xAI uses OpenAI-compatible API
                model = ChatOpenAI(
                    base_url="https://api.x.ai/v1",
                    api_key=xai_api_key,
                    model=config.model_id,
                    temperature=0,
                )
                self.models[config.name] = model
                self.model_configs[config.name] = config
                print(f"  ✓ {config.display_name}")

            except Exception as e:
                print(f"  ✗ {config.display_name}: {e}")

    def _setup_google_models(self):
        """Setup Google Gemini models"""
        if not os.getenv("GOOGLE_API_KEY"):
            print("⚠️  No Google API key found")
            return

        print("🔴 Setting up Google Gemini models...")

        gemini_models = [
            ModelConfig(
                "gemini-pro",
                ModelProvider.GOOGLE,
                "gemini-pro",
                "Gemini Pro",
                "general",
                cost_per_1k_tokens=0.0005,
                description="Google's most capable model",
            ),
            ModelConfig(
                "gemini-pro-vision",
                ModelProvider.GOOGLE,
                "gemini-pro-vision",
                "Gemini Pro Vision",
                "general",
                cost_per_1k_tokens=0.0025,
                description="Gemini with vision capabilities",
            ),
            ModelConfig(
                "gemini-1.5-pro",
                ModelProvider.GOOGLE,
                "gemini-1.5-pro",
                "Gemini 1.5 Pro",
                "general",
                cost_per_1k_tokens=0.0035,
                description="Latest Gemini Pro model",
            ),
            ModelConfig(
                "gemini-1.5-flash",
                ModelProvider.GOOGLE,
                "gemini-1.5-flash",
                "Gemini 1.5 Flash",
                "fast",
                cost_per_1k_tokens=0.00075,
                description="Fast Gemini model",
            ),
        ]

        for config in gemini_models:
            try:
                model = ChatGoogleGenerativeAI(model=config.model_id, temperature=0)
                self.models[config.name] = model
                self.model_configs[config.name] = config
                print(f"  ✓ {config.display_name}")

            except Exception as e:
                print(f"  ✗ {config.display_name}: {e}")

    def _setup_custom_models(self):
        """Setup custom/open source models via API endpoints"""
        print("🟡 Setting up custom models...")

        custom_endpoints = {
            "llama-3.1-70b": os.getenv("LLAMA_ENDPOINT"),
            "deepseek-coder": os.getenv("DEEPSEEK_ENDPOINT"),
            "qwen-2.5-72b": os.getenv("QWEN_ENDPOINT"),
            "mistral-large": os.getenv("MISTRAL_ENDPOINT"),
        }

        for model_name, endpoint in custom_endpoints.items():
            if endpoint:
                try:
                    config = ModelConfig(
                        model_name,
                        ModelProvider.CUSTOM,
                        model_name,
                        model_name.replace("-", " ").title(),
                        "custom",
                        description=f"Custom model via {endpoint}",
                    )

                    model = ChatOpenAI(
                        base_url=endpoint,
                        api_key=os.getenv(
                            f"{model_name.upper().replace('-', '_')}_API_KEY", "dummy"
                        ),
                        model=model_name,
                        temperature=0,
                    )

                    self.models[config.name] = model
                    self.model_configs[config.name] = config
                    print(f"  ✓ {config.display_name}")

                except Exception as e:
                    print(f"  ✗ {model_name}: {e}")

    def _print_model_summary(self):
        """Print summary of available models"""
        by_provider = {}
        for config in self.model_configs.values():
            provider = config.provider.value
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(config)

        print("\n📊 Model Summary:")
        for provider, configs in by_provider.items():
            by_category = {}
            for config in configs:
                if config.category not in by_category:
                    by_category[config.category] = []
                by_category[config.category].append(config.display_name)

            print(f"  {provider.upper()}: {len(configs)} models")
            for category, models in by_category.items():
                print(f"    {category}: {', '.join(models)}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models with metadata"""
        return [
            {
                "name": name,
                "display_name": config.display_name,
                "provider": config.provider.value,
                "category": config.category,
                "description": config.description,
                "supports_temperature": config.supports_temperature,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
            }
            for name, config in self.model_configs.items()
        ]

    def get_models_by_category(self) -> Dict[str, List[str]]:
        """Get models grouped by category for UI display"""
        categories = {}
        for name, config in self.model_configs.items():
            category = f"{config.provider.value.title()} - {config.category.title()}"
            if category not in categories:
                categories[category] = []
            categories[category].append(
                {
                    "value": name,
                    "label": config.display_name,
                    "description": config.description,
                }
            )
        return categories

    async def query_model(
        self, model_name: str, prompt: str, temperature: float = 0.0
    ) -> ModelResponse:
        """Query a specific model and return standardized response"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")

        model = self.models[model_name]
        config = self.model_configs[model_name]

        start_time = datetime.now()

        try:
            # Adjust temperature for models that support it
            if hasattr(model, "temperature") and config.supports_temperature:
                model.temperature = temperature

            # Query the model
            message = HumanMessage(content=prompt)
            response = await model.ainvoke([message])

            end_time = datetime.now()
            latency = int((end_time - start_time).total_seconds() * 1000)

            # Extract content
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Estimate cost if available
            cost_estimate = None
            tokens_used = None
            if config.cost_per_1k_tokens and hasattr(response, "usage_metadata"):
                tokens_used = getattr(response.usage_metadata, "total_tokens", None)
                if tokens_used:
                    cost_estimate = (tokens_used / 1000) * config.cost_per_1k_tokens

            return ModelResponse(
                model_name=model_name,
                provider=config.provider.value,
                content=content,
                metadata={
                    "temperature": temperature,
                    "supports_temperature": config.supports_temperature,
                    "category": config.category,
                },
                timestamp=start_time.isoformat(),
                latency_ms=latency,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
            )

        except Exception as e:
            return ModelResponse(
                model_name=model_name,
                provider=config.provider.value,
                content=f"Error: {str(e)}",
                metadata={"error": True, "error_type": type(e).__name__},
                timestamp=start_time.isoformat(),
                latency_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    async def query_multiple_models(
        self,
        model_names: List[str],
        prompt: str,
        temperature: float = 0.0,
        iterations: int = 1,
    ) -> List[ModelResponse]:
        """Query multiple models with the same prompt"""

        tasks = []
        for model_name in model_names:
            if model_name in self.models:
                for iteration in range(iterations):
                    tasks.append(self.query_model(model_name, prompt, temperature))
            else:
                print(f"⚠️  Model '{model_name}' not available")

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to ModelResponse
        valid_responses = []
        for response in responses:
            if isinstance(response, Exception):
                print(f"❌ Model query failed: {response}")
            else:
                valid_responses.append(response)

        return valid_responses
