# core/vlm_client_gemini.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.cache import CacheKeys, JsonCache
from utils.config import VLMConfig

# NOTE:
# This client is written to work with Google's official Python SDK:
#   pip install google-generativeai
# If you use a different Gemini client, you can swap _call_gemini() implementation.


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _read_image_bytes(path: str | Path) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p.read_bytes()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Robust JSON parsing:
    - first try direct json.loads
    - then extract the first {...} block and try again
    """
    text = text.strip()

    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object substring
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"Could not find JSON object in model output. Output was:\n{text[:2000]}")

    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception as e:
        raise ValueError(
            f"Found JSON-like block but failed to parse. Error: {e}\n"
            f"Candidate:\n{candidate[:2000]}"
        )


def _sleep_backoff(base: float, attempt: int) -> None:
    # attempt: 0,1,2,... => backoff base*(attempt+1)
    time.sleep(base * (attempt + 1))


@dataclass(frozen=True)
class ProposedFeature:
    feature: str
    description: str


@dataclass(frozen=True)
class FeatureProposalResult:
    reasoning: str
    features: List[ProposedFeature]


@dataclass(frozen=True)
class BatchSeverityResult:
    labels: Dict[str, int]  # tile_id -> severity 0/1/2
    reasoning: Optional[str] = None


class GeminiVLMClient:
    """
    Gemini client wrapper for:
      1) contrastive spurious feature proposal (Grid A + baseline grids)
      2) batch severity annotation on labeled grid tiles
      3) single-image severity annotation (optional spot-check)

    Caching:
      You can pass JsonCache instances for proposals and annotations to avoid repeated cost.
    """

    # Version strings are part of cache keys; bump if you modify prompts.
    PROMPT_VERSION_PROPOSE = "contrastive_propose_v1"
    PROMPT_VERSION_BATCH_SEVERITY = "batch_severity_v1"
    PROMPT_VERSION_SINGLE_SEVERITY = "single_severity_v1"

    def __init__(
        self,
        cfg: VLMConfig,
        proposals_cache: Optional[JsonCache] = None,
        severity_cache: Optional[JsonCache] = None,
        logger=None,
    ):
        self.cfg = cfg
        self.proposals_cache = proposals_cache
        self.severity_cache = severity_cache
        self.logger = logger

        self._model = None  # lazily initialized

    # -------------------------
    # Public API
    # -------------------------

    def propose_spurious_features_contrastive(
        self,
        dim_id: int,
        grid_a_path: str | Path,
        baseline_grid_paths: List[str | Path],
    ) -> FeatureProposalResult:
        """
        Calls Gemini with the contrastive hypothesis generation prompt.
        Returns top 3 spurious features (feature + description) + reasoning.

        Expected JSON output:
        {
          "reasoning": "...",
          "top_spurious_features": [
            {"feature": "...", "description": "..."},
            ...
          ]
        }
        """
        grid_a_path = str(Path(grid_a_path))
        baseline_grid_paths = [str(Path(p)) for p in baseline_grid_paths]

        cache_key = None
        if self.proposals_cache is not None:
            cache_key = CacheKeys.propose_features(
                dim_id=dim_id,
                grid_a_path=grid_a_path,
                grid_b_paths=baseline_grid_paths,
                prompt_version=self.PROMPT_VERSION_PROPOSE,
            )
            cached = self.proposals_cache.get(cache_key)
            if cached is not None:
                return self._deserialize_feature_proposal(cached)

        system_prompt = self._system_prompt_contrastive_propose()
        user_prompt = self._user_prompt_contrastive_propose()

        # Build multimodal parts: (text + images)
        parts: List[Any] = []
        parts.append({"text": user_prompt})
        parts.append(self._image_part(grid_a_path))
        for p in baseline_grid_paths:
            parts.append(self._image_part(p))

        raw_text = self._call_gemini(system_prompt=system_prompt, parts=parts)
        data = _safe_json_loads(raw_text)

        result = self._parse_feature_proposal_json(data)

        if self.proposals_cache is not None and cache_key is not None:
            self.proposals_cache.set(cache_key, self._serialize_feature_proposal(result))
            self.proposals_cache.save()

        return result

    def annotate_batch_severity_on_grid(
        self,
        feature_description: str,
        grid_path: str | Path,
        tile_ids: List[str],
    ) -> BatchSeverityResult:
        """
        Calls Gemini with a labeled grid image (tile IDs drawn on each tile).
        Returns tile_id -> severity (0/1/2).

        Expected strict JSON:
        {
          "feature": "...",
          "labels": {
             "img_0000": 2,
             "img_0001": 0,
             ...
          },
          "reasoning": "optional"
        }
        """
        grid_path = str(Path(grid_path))

        cache_key = None
        if self.severity_cache is not None:
            cache_key = CacheKeys.batch_severity(
                feature=feature_description,
                grid_path=grid_path,
                tile_ids=tile_ids,
                prompt_version=self.PROMPT_VERSION_BATCH_SEVERITY,
            )
            cached = self.severity_cache.get(cache_key)
            if cached is not None:
                return self._deserialize_batch_severity(cached)

        system_prompt = self._system_prompt_batch_severity()
        user_prompt = self._user_prompt_batch_severity(feature_description=feature_description, tile_ids=tile_ids)

        parts: List[Any] = [{"text": user_prompt}, self._image_part(grid_path)]
        raw_text = self._call_gemini(system_prompt=system_prompt, parts=parts)
        data = _safe_json_loads(raw_text)

        res = self._parse_batch_severity_json(data, tile_ids=tile_ids)

        if self.severity_cache is not None and cache_key is not None:
            self.severity_cache.set(cache_key, self._serialize_batch_severity(res))
            self.severity_cache.save()

        return res

    def annotate_single_severity(
        self,
        feature_description: str,
        image_path: str | Path,
        image_id: str,
    ) -> int:
        """
        Single-image severity annotation for spot-checking / resolving ambiguous batches.
        Returns severity in {0,1,2}.

        Expected strict JSON:
        { "severity": 0|1|2, "reasoning": "..." }
        """
        image_path = str(Path(image_path))

        cache_key = None
        if self.severity_cache is not None:
            cache_key = CacheKeys.single_severity(
                feature=feature_description,
                image_id=image_id,
                prompt_version=self.PROMPT_VERSION_SINGLE_SEVERITY,
            )
            cached = self.severity_cache.get(cache_key)
            if cached is not None:
                # cached is expected to be {"severity": int, ...}
                if "severity" in cached and cached["severity"] in (0, 1, 2):
                    return int(cached["severity"])

        system_prompt = self._system_prompt_single_severity()
        user_prompt = self._user_prompt_single_severity(feature_description=feature_description)

        parts: List[Any] = [{"text": user_prompt}, self._image_part(image_path)]
        raw_text = self._call_gemini(system_prompt=system_prompt, parts=parts)
        data = _safe_json_loads(raw_text)

        sev = int(data.get("severity"))
        if sev not in (0, 1, 2):
            raise ValueError(f"Invalid severity returned: {sev}. Full JSON: {data}")

        if self.severity_cache is not None and cache_key is not None:
            self.severity_cache.set(cache_key, {"severity": sev, "raw": data})
            self.severity_cache.save()

        return sev

    # -------------------------
    # Prompts
    # -------------------------

    @staticmethod
    def _system_prompt_contrastive_propose() -> str:
        return (
            "You are an expert AI vision researcher specializing in model interpretability and dataset bias.\n"
            "You will be shown:\n"
            "1) Grid A: images from a discovered latent dimension/subgroup that mixes multiple true classes.\n"
            "2) Baseline grids: representative samples of the involved true classes from the standard distribution.\n\n"
            "Your goal is to identify SPURIOUS FEATURES that cause the model to group these semantically different "
            "objects together. Spurious features are visual shortcuts (background, geometry, pose, framing, orientation, "
            "lighting, context) that are not essential to object identity.\n\n"
            "Rules:\n"
            "- Compare Grid A against the baseline grids to find what is distinctive in Grid A.\n"
            "- Do NOT use object class names as features.\n"
            "- Ignore compression/quality artifacts.\n"
            "- Be conservative: only mention features supported by visible evidence.\n"
            "- Output MUST be strict JSON only.\n"
        )

    @staticmethod
    def _user_prompt_contrastive_propose() -> str:
        return (
            "Task: Comparative spurious feature discovery.\n\n"
            "You will see multiple images. The FIRST image is Grid A (discovered dimension). "
            "The subsequent images are baseline grids for the true classes present in Grid A.\n\n"
            "Return JSON exactly in this format:\n"
            "{\n"
            '  "reasoning": "1-3 concise sentences",\n'
            '  "top_spurious_features": [\n'
            '    {"feature": "...", "description": "..."},\n'
            '    {"feature": "...", "description": "..."},\n'
            '    {"feature": "...", "description": "..."}\n'
            "  ]\n"
            "}\n\n"
            "The feature strings should be short noun phrases. "
            "Prioritize discriminative geometry/orientation/framing/background cues."
        )

    @staticmethod
    def _system_prompt_batch_severity() -> str:
        return (
            "You are a vision bias analysis assistant.\n"
            "You will be given ONE labeled grid image. Each tile has an ID text label like img_0000.\n"
            "Your task is to assign a SEVERITY score for a given visual feature for EACH tile.\n\n"
            "Severity scale:\n"
            "0 = feature clearly absent\n"
            "1 = feature weakly/partially present\n"
            "2 = feature strongly/dominantly present\n\n"
            "Rules:\n"
            "- Use only visible evidence.\n"
            "- Do not speculate.\n"
            "- Be conservative when uncertain (prefer lower severity).\n"
            "- Output MUST be strict JSON only.\n"
        )

    @staticmethod
    def _user_prompt_batch_severity(feature_description: str, tile_ids: List[str]) -> str:
        # We explicitly provide tile IDs to reduce missing keys / hallucinated keys.
        ids = ", ".join(tile_ids)
        return (
            f"Feature definition:\n{feature_description}\n\n"
            "You must label every tile in the grid.\n"
            f"Valid tile IDs are:\n{ids}\n\n"
            "Return JSON exactly in this format:\n"
            "{\n"
            '  "feature": "<repeat the feature phrase>",\n'
            '  "labels": {\n'
            '    "img_0000": 0,\n'
            '    "img_0001": 2\n'
            "  },\n"
            '  "reasoning": "optional 1-2 sentences"\n'
            "}\n\n"
            "IMPORTANT: labels must be integers 0,1,2 and keys must exactly match the tile IDs."
        )

    @staticmethod
    def _system_prompt_single_severity() -> str:
        return (
            "You are a vision bias analysis assistant.\n"
            "Your task is to evaluate the SEVERITY of a specific visual feature in a single image.\n\n"
            "Severity scale:\n"
            "0 = feature clearly absent\n"
            "1 = feature weakly/partially present\n"
            "2 = feature strongly/dominantly present\n\n"
            "Rules:\n"
            "- Use only visible evidence.\n"
            "- Do not speculate.\n"
            "- Be conservative when uncertain (prefer lower severity).\n"
            "- Output MUST be strict JSON only.\n"
        )

    @staticmethod
    def _user_prompt_single_severity(feature_description: str) -> str:
        return (
            f"Feature definition:\n{feature_description}\n\n"
            "Return JSON exactly in this format:\n"
            '{ "severity": 0, "reasoning": "1-2 concise sentences" }\n'
            "Only output JSON."
        )

    # -------------------------
    # Gemini call plumbing
    # -------------------------

    def _ensure_model(self):
        if self._model is not None:
            return

        if not self.cfg.api_key:
            raise ValueError("Gemini API key is missing. Set VLMConfig.api_key before calling Gemini.")

        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            raise ImportError(
                "google-generativeai is not installed. Install with: pip install google-generativeai\n"
                f"Original import error: {e}"
            )

        genai.configure(api_key=self.cfg.api_key)
        # The GenerativeModel accepts system_instruction in newer SDK versions.
        self._genai = genai
        self._model = genai.GenerativeModel(self.cfg.model_name)

    def _image_part(self, image_path: str) -> Any:
        """
        Create an image part for the google-generativeai SDK.
        """
        self._ensure_model()
        img_bytes = _read_image_bytes(image_path)
        # The SDK supports dict-like parts:
        return {"mime_type": "image/png", "data": img_bytes}

    def _call_avalai(self, system_prompt: str, parts: List[Any]) -> str:
        """
        Calls AvalAI (OpenAI-compatible API) with retries.
        Returns response content string.
        """

        url = "https://api.avalai.ir/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",  # <-- put AvalAI key here
        }

        # Convert your `parts` to OpenAI message format
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for p in parts:
            messages.append({"role": "user", "content": str(p)})

        payload = {
            "model": self.cfg.model_name,  # e.g. "gpt-4o"
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }

        last_err: Optional[Exception] = None

        for attempt in range(self.cfg.max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                last_err = e
                if self.logger:
                    self.logger.warning(
                        f"AvalAI call failed (attempt {attempt+1}/{self.cfg.max_retries}): {last_err}"
                    )

                time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))

        raise RuntimeError(
            f"AvalAI call failed after {self.cfg.max_retries} attempts. Last error: {last_err}"
        )

    def _call_gemini(self, system_prompt: str, parts: List[Any]) -> str:
        """
        Calls Gemini with retries.
        Returns response.text
        """
        self._ensure_model()

        # Generation config
        generation_config = {
            "temperature": self.cfg.temperature,
            "max_output_tokens": self.cfg.max_output_tokens,
        }

        last_err: Optional[Exception] = None

        for attempt in range(self.cfg.max_retries):
            try:
                # Many SDK versions accept system_instruction in generate_content.
                resp = self._model.generate_content(
                    contents=parts,
                    generation_config=generation_config,
                    system_instruction=system_prompt,
                )
                text = getattr(resp, "text", None)
                if not text:
                    # sometimes candidates exist without .text
                    text = str(resp)
                return text

            except TypeError:
                # Fallback for SDKs that don't accept system_instruction in generate_content:
                try:
                    # Recreate model with system_instruction if supported
                    self._model = self._genai.GenerativeModel(
                        self.cfg.model_name, system_instruction=system_prompt
                    )
                    resp = self._model.generate_content(
                        contents=parts,
                        generation_config=generation_config,
                    )
                    text = getattr(resp, "text", None)
                    if not text:
                        text = str(resp)
                    return text
                except Exception as e2:
                    last_err = e2
            except Exception as e:
                last_err = e

            if self.logger:
                self.logger.warning(f"Gemini call failed (attempt {attempt+1}/{self.cfg.max_retries}): {last_err}")

            _sleep_backoff(self.cfg.retry_backoff_s, attempt)

        raise RuntimeError(f"Gemini call failed after {self.cfg.max_retries} attempts. Last error: {last_err}")

    # -------------------------
    # Parsing + (de)serialization
    # -------------------------

    @staticmethod
    def _parse_feature_proposal_json(data: Dict[str, Any]) -> FeatureProposalResult:
        reasoning = str(data.get("reasoning", "")).strip()

        feats_raw = data.get("top_spurious_features", [])
        if not isinstance(feats_raw, list) or len(feats_raw) == 0:
            raise ValueError(f"Invalid top_spurious_features in response JSON: {data}")

        features: List[ProposedFeature] = []
        for item in feats_raw[:3]:
            if not isinstance(item, dict):
                continue
            f = str(item.get("feature", "")).strip()
            d = str(item.get("description", "")).strip()
            if f:
                features.append(ProposedFeature(feature=f, description=d))

        if len(features) == 0:
            raise ValueError(f"No valid features parsed from response JSON: {data}")

        # ensure exactly 3 if possible
        if len(features) < 3:
            # pad with empty descriptions if missing
            while len(features) < 3:
                features.append(ProposedFeature(feature="(missing)", description=""))

        return FeatureProposalResult(reasoning=reasoning, features=features[:3])

    @staticmethod
    def _parse_batch_severity_json(data: Dict[str, Any], tile_ids: List[str]) -> BatchSeverityResult:
        labels = data.get("labels", None)
        if not isinstance(labels, dict):
            raise ValueError(f"Invalid 'labels' field in batch severity JSON: {data}")

        out: Dict[str, int] = {}
        for tid in tile_ids:
            if tid not in labels:
                # allow missing; we will error after loop
                continue
            v = labels[tid]
            try:
                iv = int(v)
            except Exception:
                continue
            if iv not in (0, 1, 2):
                continue
            out[tid] = iv

        missing = [tid for tid in tile_ids if tid not in out]
        if missing:
            raise ValueError(
                f"Batch severity JSON missing/invalid for {len(missing)} tiles. Missing examples: {missing[:5]}\n"
                f"Full JSON: {data}"
            )

        reasoning = data.get("reasoning", None)
        if reasoning is not None:
            reasoning = str(reasoning).strip()

        return BatchSeverityResult(labels=out, reasoning=reasoning)

    @staticmethod
    def _serialize_feature_proposal(res: FeatureProposalResult) -> Dict[str, Any]:
        return {
            "reasoning": res.reasoning,
            "top_spurious_features": [{"feature": f.feature, "description": f.description} for f in res.features],
        }

    @staticmethod
    def _deserialize_feature_proposal(obj: Dict[str, Any]) -> FeatureProposalResult:
        return GeminiVLMClient._parse_feature_proposal_json(obj)

    @staticmethod
    def _serialize_batch_severity(res: BatchSeverityResult) -> Dict[str, Any]:
        return {"labels": res.labels, "reasoning": res.reasoning}

    @staticmethod
    def _deserialize_batch_severity(obj: Dict[str, Any]) -> BatchSeverityResult:
        labels = obj.get("labels", {})
        reasoning = obj.get("reasoning", None)
        if not isinstance(labels, dict):
            raise ValueError(f"Invalid cached batch severity object: {obj}")
        # ensure ints
        fixed = {k: int(v) for k, v in labels.items()}
        return BatchSeverityResult(labels=fixed, reasoning=reasoning)