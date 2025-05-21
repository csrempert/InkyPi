"""
ai_storyteller.py  –  InkyPi standalone plugin
(compatible with openai-python ≥ 1.0)

Creates: illustration + story + narration.
"""

from __future__ import annotations
import logging, subprocess, tempfile, requests, pathlib, os
from typing import Any, Dict
from PIL import Image
from io import BytesIO

# ── openai v1 client ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI          # pip install openai>=1.0
except ImportError as exc:
    raise RuntimeError(
        "openai>=1.0 is required.  Run:  pip install openai --upgrade"
    ) from exc

# InkyPi base
from plugins.base_plugin.base_plugin import BasePlugin

logger = logging.getLogger(__name__)


class AIStoryteller(BasePlugin):
    """Create illustration + story + audio for bedtime."""

    # ─────────────────────────────────── public hook ──────────────────────────
    def generate_image(  # type: ignore[override]
        self, settings: Dict[str, Any], device_config: "DeviceConfig"
    ) -> Image.Image:

        prompt       = str(settings.get("textPrompt", "")).strip()
        text_model   = settings.get("textModel", "gpt-4o")
        story_len    = settings.get("storyLength", "medium")
        image_model  = settings.get("imageModel", "dall-e-3")
        quality      = settings.get("quality", "standard")
        voice_id     = settings.get("voiceId", "21m00Tcm4TlvDq8ikWAM")

        if not prompt:
            raise RuntimeError("Storyteller: A prompt is required.")

        api_key = device_config.load_env_key("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Storyteller: OPENAI_API_KEY is missing.")
        client = OpenAI(api_key=api_key)

        # 1️⃣  story
        story_text = self._create_story(client, prompt, text_model, story_len)
        # 2️⃣  narration
        self._narrate(story_text, device_config, voice_id)
        # 3️⃣  illustration
        illustration = self._create_image(client, prompt, image_model, quality)

        return illustration

    # ───────────────────────────── settings template ──────────────────────────
    def generate_settings_template(self) -> Dict[str, Any]:
        return {
            "textPrompt": {
                "type": "text",
                "label": "Story prompt",
                "placeholder": "e.g. A boy and his cat visit the moon",
                "default": "",
            },
            "textModel":   {"type": "text", "default": "gpt-4o"},
            "storyLength": {"type": "text", "default": "medium"},
            "imageModel":  {"type": "text", "default": "dall-e-3"},
            "quality":     {"type": "text", "default": "standard"},
            "voiceId":     {"type": "text", "default": "21m00Tcm4TlvDq8ikWAM"},
        }

    # ───────────────────────────── helpers ────────────────────────────────────
    def _create_story(
        self, client: OpenAI, theme_prompt: str, model: str, length: str
    ) -> str:
        system_prompt = (
            "You are a warm, gentle storyteller for toddlers. "
            "Write a calming bedtime story (~250 words) featuring a toddler "
            "and his cat. End happily."
        )
        user_prompt   = f"Theme: {theme_prompt}"
        max_toks      = {"short": 200, "medium": 400, "long": 600}.get(length, 400)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_toks,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.exception("OpenAI ChatCompletion error")
            raise RuntimeError(f"Storyteller: Story generation failed: {exc}") from exc

    # ------------------------------------------------------------------------ #
    def _create_image(
        self, client: OpenAI, theme_prompt: str, model: str, quality: str
    ) -> Image.Image:
        image_prompt = (
            "Cute, colorful storybook illustration, soft watercolor style, "
            "featuring a toddler and his cat, "
            f"theme: {theme_prompt}, gentle night-time lighting."
        )
        try:
            if model in ("dall-e-3", "gpt-4o"):
                params = {
                    "model":  model,
                    "prompt": image_prompt,
                    "n":      1,
                    "size":   "1792x1024",
                }
                if model == "dall-e-3":
                    params["quality"] = quality
                img_resp = client.images.generate(**params)
            elif model == "dall-e-2":
                img_resp = client.images.generate(
                    model="dall-e-2",
                    prompt=image_prompt,
                    n=1,
                    size="1792x1024",
                )
            else:
                raise RuntimeError(f"Unsupported image model: {model}")

            url = img_resp.data[0].url
            img_bytes = requests.get(url, timeout=30).content
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            logger.exception("OpenAI image generation error")
            raise RuntimeError(f"Storyteller: Illustration failed: {exc}") from exc

    # ───────────────────────────── narration ─────────────────────────────────
    def _narrate(self, story: str, config: "DeviceConfig", voice_id: str) -> None:
        if config.load_env_key("ELEVENLABS_API_KEY"):
            self._narrate_eleven_labs(story, config, voice_id)
        else:
            self._narrate_espeak(story, voice_id)

    def _narrate_espeak(self, story: str, _: str) -> None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav = tmp.name
            subprocess.run(["espeak", "-ven+f3", "-s150", "-w", wav, story], check=True)
            subprocess.Popen(["aplay", wav])
        except Exception as exc:
            raise RuntimeError(f"Storyteller: espeak failed: {exc}") from exc

    def _narrate_eleven_labs(
        self, story: str, config: "DeviceConfig", voice_id: str
    ) -> None:
        import requests as _rq

        api_key = config.load_env_key("ELEVENLABS_API_KEY")
        url     = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": story,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.3, "similarity_boost": 0.85},
        }
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        try:
            r = _rq.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code in (401, 403):
                logger.warning("ElevenLabs auth/voice error (%s) – falling back to eSpeak", r.status_code)
                self._narrate_espeak(story, voice_id)
                return
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(r.content)
                mp3 = tmp.name
            subprocess.Popen(["mpg123", mp3])
        except Exception as exc:
            raise RuntimeError(f"Storyteller: ElevenLabs TTS failed: {exc}") from exc