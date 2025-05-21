"""
ai_storyteller.py  –  InkyPi standalone plugin

Fully self‑contained bedtime‑story generator:

1. Generates an illustration with OpenAI Images (DALL·E 3)
2. Generates a short bedtime story with ChatGPT
3. Narrates the story out loud (ElevenLabs if API key present, else eSpeak)
4. Returns the PIL.Image for the Inky Impression display

Environment variables required (set in Device → Env in InkyPi):

OPENAI_API_KEY          : for both image + story generation
Optional:
ELEVENLABS_API_KEY      : high‑quality TTS (defaults to eSpeak if absent)
STORYTELLER_VOICE_ID    : ElevenLabs voice ID (defaults Unknown)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from io import BytesIO
from typing import Any, Dict

import requests
from PIL import Image

# InkyPi base
from plugins.base_plugin.base_plugin import BasePlugin

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("openai package not installed.  Run `pip install openai`.") from exc

logger = logging.getLogger(__name__)


class AIStoryteller(BasePlugin):
    """Create illustration + story + audio."""

    # ------------------------------------------------------------------ #
    #  Public InkyPi hooks
    # ------------------------------------------------------------------ #

    def generate_image(  # type: ignore[override]
        self, settings: Dict[str, Any], device_config: "DeviceConfig"
    ) -> Image.Image:
        """
        Full pipeline executed by InkyPi:
            • read prompt from settings["textPrompt"]
            • create story  (GPT‑4o)
            • create image  (DALL·E 3)
            • narrate story (ElevenLabs or eSpeak)
            • return PIL.Image for display
        """
        prompt: str = str(settings.get("textPrompt", "")).strip()
        if not prompt:
            raise RuntimeError("Storyteller: A prompt is required.")

        text_model   = settings.get("textModel", "gpt-4o")
        story_len    = settings.get("storyLength", "medium")
        image_model  = settings.get("imageModel", "dall-e-3")
        quality      = settings.get("quality", "standard")
        voice_id     = settings.get("voiceId", "Unknown")

        api_key: str | None = device_config.load_env_key("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Storyteller: OPENAI_API_KEY is missing.")
        openai.api_key = api_key

        # 1️⃣  Build story
        story_text = self._create_story(prompt, text_model, story_len)

        # 2️⃣  Narrate asynchronously (non‑blocking)
        self._narrate(story_text, device_config, voice_id)

        # 3️⃣  Build illustration
        illustration = self._create_image(prompt, image_model, quality)

        return illustration

    # ------------------------------------------------------------------ #
    #  Settings UI (minimal)
    # ------------------------------------------------------------------ #

    def generate_settings_template(self) -> Dict[str, Any]:
        """Expose a single textPrompt field in the InkyPi UI."""
        return {
            "textPrompt": {
                "type": "text",
                "label": "Story prompt",
                "placeholder": "e.g. A boy and his dog visit the moon",
                "default": "",
            },
            "textModel": {
                "type": "text",
                "default": "gpt-4o",
            },
            "storyLength": {
                "type": "text",
                "default": "medium",
            },
            "imageModel": {
                "type": "text",
                "default": "dall-e-3",
            },
            "quality": {
                "type": "text",
                "default": "standard",
            },
            "voiceId": {
                "type": "text",
                "default": "Unknown",
            },
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    # ------ STORY ----------------------------------------------------- #
    def _create_story(self, theme_prompt: str, model: str, length: str) -> str:
        """Generate a gentle, ~250‑word bedtime story."""
        system_prompt = (
            "You are a warm, gentle storyteller for toddlers. "
            "Write a calming bedtime story (~250 words) featuring a toddler "
            "and his dog. End happily."
        )
        user_prompt = f"Theme: {theme_prompt}"
        length_map = {"short": 200, "medium": 400, "long": 600}
        max_toks = length_map.get(length, 400)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_toks,
                temperature=0.7,
            )
            story = response.choices[0].message.content.strip()
            logger.debug("Story generated (%d chars)", len(story))
            return story
        except Exception as exc:  # pragma: no cover
            logger.exception("OpenAI ChatCompletion error")
            raise RuntimeError(f"Storyteller: Story generation failed: {exc}") from exc

    # ------ ILLUSTRATION --------------------------------------------- #
    def _create_image(self, theme_prompt: str, model: str, quality: str) -> Image.Image:
        """Create a cute illustration with DALL·E 3 and return as PIL.Image."""
        image_prompt = (
            "Cute, colorful storybook illustration, soft watercolor style, "
            "featuring a toddler and his dog, "
            f"theme: {theme_prompt}, gentle night‑time lighting."
        )
        try:
            if model in ("dall-e-3", "gpt-4o"):
                params = {
                    "model": model,
                    "prompt": image_prompt,
                    "n": 1,
                    "size": "1024x768",
                }
                if model == "dall-e-3":
                    params["quality"] = quality
                img_resp = openai.images.generate(**params)
            elif model == "dall-e-2":
                img_resp = openai.images.generate(
                    model="dall-e-2",
                    prompt=image_prompt,
                    n=1,
                    size="1024x768",
                )
            else:
                raise RuntimeError(f"Unsupported image model: {model}")
            image_url = img_resp.data[0].url
            img_bytes = requests.get(image_url, timeout=30).content
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            logger.exception("OpenAI image generation error")
            raise RuntimeError(f"Storyteller: Illustration failed: {exc}") from exc

    # ---------------------- AUDIO / TTS ------------------------------ #
    def _narrate(self, story: str, device_config: "DeviceConfig", voice_id: str) -> None:
        """Speak story using ElevenLabs if available; else fallback to eSpeak."""
        if device_config.load_env_key("ELEVENLABS_API_KEY"):
            self._narrate_eleven_labs(story, device_config, voice_id)
        else:
            self._narrate_espeak(story, voice_id)

    # ---- espeak fallback ------------------------------------------- #
    def _narrate_espeak(self, story: str, voice_id: str) -> None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            subprocess.run(
                ["espeak", "-ven+f3", "-s150", "-w", wav_path, story],
                check=True,
            )
            subprocess.Popen(["aplay", wav_path])
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Storyteller: espeak not installed. `sudo apt install espeak-ng`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Storyteller: espeak failed: {exc}") from exc

    # ---- ElevenLabs high‑quality voice ----------------------------- #
    def _narrate_eleven_labs(self, story: str, device_config: "DeviceConfig", voice_id: str) -> None:
        import requests as _rq

        api_key = device_config.load_env_key("ELEVENLABS_API_KEY")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": story,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.3, "similarity_boost": 0.85},
        }
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}

        try:
            r = _rq.post(url, json=payload, timeout=30)
            r.raise_for_status()
            audio_bytes = r.content
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_bytes)
                mp3_path = tmp.name
            subprocess.Popen(["mpg123", mp3_path])
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Storyteller: mpg123 not installed. `sudo apt install mpg123`."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Storyteller: ElevenLabs TTS failed: {exc}") from exc