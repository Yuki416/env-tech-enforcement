"""端到端煙霧測試：跑一支影片，看 Co-DETR litter 來源是不是真的接上了。

在 enforcement 容器內執行。設 ENFORCE_CODETR_URL 才會啟用 Co-DETR；不設就是原本的 YOLO，
所以同一支腳本可以跑兩次做 A/B。

    docker compose exec -d -e ENFORCE_CODETR_URL=http://codetr:8091 enforcement \\
      bash -c 'cd /app && python codetr_service/smoke_e2e.py --video /testset/xxx.avi \\
                 --out /output/_smoke_codetr > /tmp/smoke_codetr.log 2>&1'
    docker compose exec enforcement tail -f /tmp/smoke_codetr.log
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-calls", type=int, default=60)
    args = parser.parse_args()

    from enforcement_app.config import load_config
    from enforcement_app.models import VideoJob
    from enforcement_app.pipeline_adapter import PipelineEngine, build_job_options

    url = os.getenv("ENFORCE_CODETR_URL", "").strip()
    print(f"=== Co-DETR litter 來源：{url or '(未啟用，走 YOLO)'} ===", flush=True)

    import dataclasses

    config = dataclasses.replace(load_config(), output_root=Path(args.out))
    engine = PipelineEngine(config)
    engine.warmup()

    job = VideoJob(
        job_id="smoke",
        video_path=Path(args.video),
        options=build_job_options(
            enable_litter=True,
            enable_urination=True,
            require_plate_for_csv=False,
            heavy_frame_stride=args.stride,
            max_vlm_calls_per_video=args.max_calls,
        ),
    )

    t0 = time.monotonic()
    done = engine.process_video(job, lambda _u: None)
    elapsed = time.monotonic() - t0

    print(f"\n=== 完成：{elapsed:.1f}s，violations={done.violations} ===", flush=True)
    audit = Path(done.output_dir) / "vlm_decisions.csv" if done.output_dir else None
    if audit and audit.is_file():
        with audit.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        kinds = Counter(r["event_type"] for r in rows)
        print(f"vlm_decisions.csv：{len(rows)} 列")
        for k, v in kinds.most_common():
            print(f"  {k:<24} {v}")
        behaviors = Counter(
            r["vlm_behavior"] for r in rows if r.get("event_type") == "vlm_call"
        )
        print("VLM 判定分佈：", dict(behaviors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
