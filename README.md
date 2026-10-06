**English** | [中文](README.zh-TW.md)

# Environmental Tech Enforcement: Detecting Illegal Littering in CCTV Footage

Littering enforcement relies heavily on CCTV, but someone still has to watch the footage to find the violations.
This project trains a detector that finds **littering in progress** in fixed roadside CCTV video, so inspectors only review the clips it flags.

This repository contains the full implementation: dataset construction, the training pipeline, evaluation, a YOLO baseline, and the enforcement system the model is deployed in.
**Model weights and data are not included.** The footage was provided by a local environmental protection bureau and contains license plates and people.

![Thrown vs. landed litter](docs/figs/fig_dyn_static.png)
*The same piece of litter in one test clip: the moment it is thrown (19×16 px) and after it lands (15×17 px), on a 2592×1944 frame.*

## Task and data

- **Source:** fixed roadside CCTV, 2592×1944, 10 fps. Violation segments are cut into short clips.
- **Classes (5):** car, motorcycle, person, **litter** (small items: cigarette butts, PET bottles, boxes), and **garbage** (large items: garbage bags, suitcases).
- **Dataset:** 122 clips, 9,349 frames, 51,816 boxes, all annotated frame by frame. 5,467 of the boxes are litter and 1,263 are garbage.
- **Splits:** by clip (train 99 / val 15 / test 8), so no clip appears in two splits.
- **Why it is hard:** after resizing to a 1024-px long side, the median litter box is about **11×13 px**.

## Method: four iterations

The approach is iterative. Define the problem precisely, build a solution, and measure it with a suitable metric. When it fails, find out which step failed and why before changing anything.
The project ran 18 training rounds plus a YOLO baseline. The four turning points were:

**1. Make the model learn the training set first.**
Round 6 detected only 7% of litter *on the training set itself*. Round 7 combined three fixes: oversampling images that contain litter, multiplying the litter and garbage classification loss by 5, and using separate learning rates for backbone and head. Training-set miss rate fell from 93% to 15%.
From then on, every data change was first checked on the training set.

**2. Redefine the problem: which "litter" do we need to detect?**
Litter labeled by another student moves more than 20 px/frame in 71–80% of cases. In our test set, **84% of litter barely moves** (median displacement 2.4 px).
Enforcement needs the *act* of throwing, not trash already lying on the ground. So evaluation was split into two benchmarks:

| Test set | Contents | What it measures |
|---|---|---|
| `dynamic166` | 166 frames, before the litter lands | Can the model catch litter in flight? |
| `static594` | 594 frames | Is there litter in the frame? |

One test clip shares a scene with a training clip (7 minutes apart), so it is excluded from both benchmarks.
Training only on dynamic frames helped at first (round 13). Making the dynamic data larger and purer (rounds 15–17) then made the static benchmark collapse. Rather than try a fourth data recipe, I stopped retraining and diagnosed the cause.

**3. Tell the model where things move: a background-subtraction changemap as a 4th input channel.**
- **First try (round 11):** worse than RGB. The official config freezes the first conv layer, which is exactly where the new channel's weights live.
- **After unfreezing (round 12):** small-object AP went from 0.000 to 0.388.
- **Why static litter suffers (round 14):** on our data, 69% of static-litter boxes were completely black in the changemap. Reading the OpenCV source showed why. In the KNN background subtractor, `learningRate=0` means *update every frame*, not *freeze*, so a static object is absorbed into the background after 5–6 updates.
- **Clean single-variable comparison (round 18):** identical to round 16 except for the changemap and the unfrozen first layer.

**4. Is Co-DETR the right architecture? A fair comparison with YOLO26.**
YOLO26l was trained on the same splits and classes with 3 seeds. Both models are scored by the same evaluation script.
On the validation set, YOLO is far behind (litter AP@0.5: 0.141 vs 0.299).
A plausible explanation is YOLO's grid assignment: about 11% of litter boxes contain no grid point. A rescaling experiment (512–1536 px) **rejected** this as the main cause. YOLO's maximum recall stays at 0.47–0.51× of Co-DETR's at every scale.

## Results

AP@0.5 on the test sets, all scored by the same evaluation script:

| Model | dynamic litter | dynamic garbage | static litter | static garbage | ms / image |
|---|---|---|---|---|---|
| Round 16 Co-DETR (RGB) | 0.062 | 0.358 | 0.109 | 0.092 | 121.9 |
| **Round 18 Co-DETR (RGB + changemap)** | **0.301** | **0.371** | **0.226** | 0.062 | — |
| YOLO26l (RGB, mean of 3 seeds, min–max) | 0.058 (0.035–0.078) | 0.183 (0.099–0.293) | 0.124 (0.042–0.195) | 0.031 (0.021–0.048) | 33.8 |

Latency is measured on an RTX 4090 in FP32. Round 18 also needs to compute the changemap, which is not included.

![AP bar chart](docs/figs/fig_ap_bar.png)

**How reliable is the gain?** Test-set litter comes from only 3 clips.
- **Frame-level bootstrap:** the round 18 − round 16 difference has a 95% CI of [0.16, 0.33].
- **Clip-level bootstrap:** the 95% CI is [−0.08, 0.88], which crosses 0.

Round 18 is clearly better on this test set, but there are too few clips to claim it generalizes to new scenes.
On static-only frames, the gain comes from fewer false positives. This is consistent with the changemap absorbing static objects.

![Round 16 vs 18 on one frame](docs/figs/fig_bgs_detect.png)

## The system

Round 18 runs as a standalone inference service inside an enforcement system. The system takes CCTV video, flags clips that may contain littering, and lets inspectors review and annotate them in a web UI.

![Web UI player](docs/figs/fig_ui_player.png)
![Web UI events](docs/figs/fig_ui_events.png)

**Online changemaps.** The system sends the model only sparse frames, but a changemap needs consecutive frames to build its background. The service therefore opens the video itself, builds the background from the start, and picks out the requested frames.
On the 8 test clips, this matches the offline pipeline exactly (44 vs 44 true positives).

**Frame stride.** The original stride of 35 frames is far too large: litter is airborne for a median of only 3 frames.
A lower score threshold is more efficient than a denser stride:

| Setting | Event hit rate | False positives | Compute |
|---|---|---|---|
| Threshold 0.40, every 2nd frame | 66.7% | 38% fewer | 43% less |
| Threshold 0.50, every frame | 66.7% | baseline | baseline |

## Repository layout

```
co-detr/                        Co-DETR training pipeline (MMDetection 3, Docker)
  configs/                      one config per training round
    co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py   round 16 (RGB)
    co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py         round 18 (RGB + changemap)
  hooks/                        MMDetection extensions
    load_changemap.py           loads the changemap PNG as the 4th channel
    data_preprocessor_4ch.py    4-channel normalization / batching
    conv1_ch4_norm_logger.py    tracks whether the 4th-channel weights actually learn
    weighted_focal_loss.py      per-class loss weights (round 7)
  experiments/NNth_vMM_*/       per-round scripts: data prep, dry run, training, inference, post-analysis
    16th_v17_own_dynamic_122vids/資料前處理/   dynamic-interval extraction (analyze_dynamic_intervals.py, build_v17_*)
    18th_v18_own_dynamic_bgs_122vids/
      資料前處理/gen_changemaps_v18.py          pre-roll KNN changemap generation
      資料前處理/build_alignment_v18.py         aligns dataset frames to source video frames
      run_dual_track_eval.sh, metrics_from_preds.py   dynamic166 / static594 evaluation
  tools/                        shared analysis scripts (error analysis, video classification, plots)
yolo-baseline/                  YOLO26l baseline: conversion, training, end-to-end eval,
                                bootstrap CIs (bootstrap_ci.py), scale sweep (scale_sweep_*.py),
                                grid-assignment analysis (positive_slots.py), speed benchmarks
enforcement-system/
  codetr_service/server.py      round 18 inference service (online changemap)
  codetr_service/verify_lr0*.py the KNN learningRate=0 finding
  codetr_service/measure_*.py   frame-stride / threshold experiments
  webui/                        web UI (stdlib http.server, no extra dependencies)
  enforcement_app/              pipeline: candidate frames → detection → event merging → evidence
```

Experiment folders are numbered by training round (`NNth`). Dataset versions are numbered separately (`vMM`).
Earlier rounds (05–15, 17) are kept as a record of the iteration and are not maintained.
Code comments are in Traditional Chinese. Some comments in early scripts describe hypotheses that were later revised. For example, `learningRate=0` is described as "freezing" the background; see iteration 3.

## Running

All training and evaluation run in Docker. Paths in the scripts are written as `/path/to/data` (datasets and results) and `/path/to/repo`; replace them with your own.

```bash
# Co-DETR image (PyTorch 2.1, MMCV 2.1, MMDetection 3.x with projects/CO-DETR)
docker build -t co-detr:latest co-detr/

# Round 18 training (2 GPUs, DDP)
bash co-detr/experiments/18th_v18_own_dynamic_bgs_122vids/train_18th_v18_bgs_4ch_ddp.sh

# Dual-track evaluation of round 16 and round 18
bash co-detr/experiments/18th_v18_own_dynamic_bgs_122vids/run_dual_track_eval.sh

# Enforcement system: Co-DETR service + Ollama + app container
cd enforcement-system && cp .env.example .env   # fill in the paths
docker compose up -d
bash webui/webui.sh start   # web UI on 127.0.0.1:8080
bash webui/webui.sh mock    # or: UI only with mock data, no models needed
```

**Expected data format.** COCO-format JSON per split. For 4-channel models, each image needs a binary changemap PNG with the same file name under `{split}_changemap/`:

```
<dataset>/
  annotations/{train,valid,test}.json, test_dynamic166.json, test_static594.json
  {train,valid,test}/<clip>_frame_NNNN.jpg
  {train,valid,test}_changemap/<clip>_frame_NNNN.png    # 0 / 255
```

## Next steps

- **Multi-object tracking (MOTIP) to filter false positives.** Keep only trajectories that look like a throw: appear next to a person or vehicle, move, then land.
- **Speed.** At 121.9 ms per image plus changemap computation, the system cannot yet keep up with a 10 fps stream in real time.
- **A vision-language model for behavior.** Recognize the throwing action itself, and violations without an object, such as public urination.

## Credits

Annotation was done jointly with my project partner, who also built the license-plate recognition (not part of this repository).
Built on [MMDetection](https://github.com/open-mmlab/mmdetection) / [Co-DETR](https://github.com/Sense-X/Co-DETR) (Apache-2.0) and [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; used only as an installed package for the baseline).

## License

[MIT](LICENSE)
