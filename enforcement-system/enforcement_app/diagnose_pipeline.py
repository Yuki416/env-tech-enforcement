"""
Diagnostic script for litter detection pipeline troubleshooting.

This script helps identify why violations are being reported as 0.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Setup logging to see all DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_ollama_service() -> bool:
    """Check if Ollama service is running and accessible."""
    logger.info("=" * 60)
    logger.info("CHECKING OLLAMA SERVICE")
    logger.info("=" * 60)

    try:
        # Try to reach Ollama API
        result = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:11434/api/tags"],
            capture_output=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.error("✗ Ollama service NOT RESPONDING")
            logger.error(f"  Error: {result.stderr.decode()}")
            return False

        # Parse the response
        try:
            data = json.loads(result.stdout.decode())
            models = data.get("models", [])
            logger.info(f"✓ Ollama service is RUNNING")
            logger.info(f"  Available models: {len(models)}")

            # Check for required model
            glm_models = [m for m in models if "glm" in m.get("name", "").lower()]
            if glm_models:
                logger.info(f"  ✓ GLM models found:")
                for m in glm_models:
                    logger.info(f"    - {m.get('name')}")
                return True
            else:
                logger.error("  ✗ No GLM models found")
                logger.error("    Expected: sparksammy/glm-4.6v-flash-unsloth")
                logger.error("    Run: ollama pull sparksammy/glm-4.6v-flash-unsloth")
                return False

        except json.JSONDecodeError:
            logger.error("✗ Ollama response invalid JSON")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ Ollama service TIMEOUT (not responding)")
        return False
    except Exception as e:
        logger.error(f"✗ Error checking Ollama: {e}")
        return False


def check_yolo_model() -> bool:
    """Check if YOLO prefilter model exists."""
    logger.info("=" * 60)
    logger.info("CHECKING YOLO PREFILTER MODEL")
    logger.info("=" * 60)

    # Check for YOLO model paths
    possible_paths = [
        ROOT / "models/yolo/runs/detect/train_person_vehicle_litter/weights/best.pt",
        ROOT / "models/yolo/weights/best.pt",
        ROOT / "models/yolo_person_vehicle.pt",
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"✓ YOLO model found: {path}")
            return True
        else:
            logger.debug(f"  Model not at: {path}")

    logger.error("✗ YOLO prefilter model NOT FOUND")
    logger.error(f"  Checked: {[str(p) for p in possible_paths]}")
    return False


def test_vlm_inference() -> bool:
    """Test if VLM can be called successfully.

    A real call is made to the configured VLM. We assert:
      (a) GLM_MODEL is actually present in the backend's model list
      (b) the call returns a non-empty response (silent 404/empty-body would
          previously be reported as PASS — see vendor 4/27 which ran with a
          missing model name and got 0 violations across 13 clips)
      (c) the round-trip is fast enough to imply GPU offload (CPU-bound ollama
          on the vendor's machine took 47s for a trivial reply, ~3 tok/s)
    """
    logger.info("=" * 60)
    logger.info("TESTING VLM INFERENCE")
    logger.info("=" * 60)

    try:
        import json
        import time

        import numpy as np

        from enforcement_app.config import load_config
        from enforcement_app.vlm_enforcement import GLMFlashClient

        cfg = load_config()
        logger.info(
            "Attempting VLM call: provider=%s model=%s base=%s",
            cfg.vlm_provider,
            cfg.glm_model,
            cfg.glm_api_base,
        )

        # (a) For ollama, verify model is actually loaded — otherwise every
        # /api/chat call returns 404 and read_text_from_image() silently
        # returns "" (which the previous version of this check accepted).
        if cfg.vlm_provider == "ollama":
            try:
                tags_result = subprocess.run(
                    ["curl", "-s", f"{cfg.glm_api_base}/api/tags"],
                    capture_output=True,
                    timeout=5,
                )
                tags = json.loads(tags_result.stdout.decode())
                names = {m.get("name") for m in tags.get("models", [])}
                if cfg.glm_model not in names:
                    logger.error(
                        "✗ Configured GLM_MODEL=%r is NOT loaded in ollama at %s",
                        cfg.glm_model,
                        cfg.glm_api_base,
                    )
                    logger.error(
                        "  Available models: %s",
                        sorted(n for n in names if n) or "<none>",
                    )
                    logger.error(
                        "  Fix: ollama pull %s   OR   export GLM_MODEL=<one of the above>",
                        cfg.glm_model,
                    )
                    return False
            except Exception as exc:
                logger.error("✗ Could not list ollama models: %s", exc)
                return False

        client = GLMFlashClient(
            api_key=cfg.glm_api_key,
            model=cfg.glm_model,
            api_base=cfg.glm_api_base,
            timeout_sec=cfg.glm_timeout_sec,
            provider=cfg.vlm_provider,
            retry_count=0,
            retry_backoff_sec=0.0,
        )

        # 64x64 solid-black BGR frame — enough for the VLM to decode without
        # triggering "image too small" rejections.
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        t0 = time.time()
        text = client.read_text_from_image(
            dummy,
            "Reply with the single word: OK",
        )
        elapsed = time.time() - t0

        # (b) empty response is a silent failure mode (404, missing model, etc.)
        if not text or not text.strip():
            logger.error(
                "✗ VLM returned empty response after %.1fs — likely model not found "
                "or connection refused. Check %s/api/tags and GLM_MODEL.",
                elapsed,
                cfg.glm_api_base,
            )
            return False

        # (c) timing sanity — a trivial 1-token reply should never take 30s+
        # on a healthy GPU-backed setup. CPU inference is the main culprit.
        if elapsed > 30.0:
            logger.warning(
                "⚠ VLM call took %.1fs for a trivial prompt — likely CPU-bound. "
                "Run tools/check_ollama_gpu.sh to verify GPU offload.",
                elapsed,
            )

        logger.info("✓ VLM call succeeded in %.2fs", elapsed)
        logger.info("  Response (first 120 chars): %r", text[:120])
        return True

    except Exception as e:
        logger.error(f"✗ VLM inference failed: {e}")
        import traceback

        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        return False


def main() -> int:
    """Run all diagnostics."""
    logger.info("\n" + "=" * 60)
    logger.info("LITTER DETECTION PIPELINE DIAGNOSTIC")
    logger.info("=" * 60 + "\n")

    results = {
        "Ollama Service": check_ollama_service(),
        "YOLO Model": check_yolo_model(),
        "VLM Inference": test_vlm_inference(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{check:.<40} {status}")

    all_passed = all(results.values())
    logger.info("=" * 60)

    if all_passed:
        logger.info("✓ All diagnostics passed! Pipeline should work.")
        logger.info("  If still seeing 0 violations, check:")
        logger.info("    - Test video contains actual violations")
        logger.info("    - VLM confidence threshold (default 0.65)")
        logger.info("    - Run with LOG_LEVEL=DEBUG to see detailed logs")
        return 0
    else:
        logger.error("✗ Diagnostics failed. Fix issues above before proceeding.")
        logger.error("\nTroubleshooting tips:")
        if not results["Ollama Service"]:
            logger.error("  1. Start Ollama service:")
            logger.error("     ollama serve")
            logger.error("  2. In another terminal, pull the model:")
            logger.error("     ollama pull sparksammy/glm-4.6v-flash-unsloth")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
