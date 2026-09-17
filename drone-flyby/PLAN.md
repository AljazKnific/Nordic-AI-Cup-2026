# Drone Flyby Implementation Plan

This plan takes the system from a working local baseline to a publicly reachable endpoint. Work in `drone-flyby/` unless a task says otherwise.

## Success criteria

- Every response is protocol-valid and echoes `request_id` and `frame` unchanged.
- Predictions use source-frame-global coordinates, including detections carried from earlier views.
- Camera commands are legal under the constraints received in each request.
- Local oracle validation prints `1.000`.
- The real implementation runs through the local evaluator without exceptions and stays within the realtime budget.
- The same tested server is reachable from outside the local network at `/predict`.

## Latest local validation

- Oracle scorer: `1.000`.
- Offline endpoint: `1.000` mAP@0.50, 25/25 responses accepted, 0 camera refusals.
- Realtime endpoint: `0.844` mAP@0.50, 25/25 frames delivered, 0 timeouts or invalid responses.
- Realtime latency: 35 ms mean, 34 ms median, 51 ms maximum.
- Validation used the optimized endpoint on port `9057`; the production port remains `9053`.
- The Docker image now includes `src/` because the live detector loads reference data at startup.
- Before a remote retry, rebuild/restart the deployed copy and confirm its per-request latency logs; Docker image build was not run locally because Docker Desktop was unavailable.
- Template matching is motion-guided on zoomed views, reducing each request from roughly 250 ms to 35 ms locally while preserving `1.000` offline mAP.

## Stage 0: Establish the local baseline

**Goal:** prove the environment, endpoint, and evaluator agree before changing model behavior.

- [x] Create or activate a Python environment (`.venv/`).
- [x] Install the dependencies from `requirements.txt`.
- [x] Start the server (the local process was already serving on port `9053`):

  ```bash
  cd drone-flyby
  .venv/bin/python api.py
  ```

- [x] In a second terminal, verify the health endpoint:

  ```bash
  curl http://localhost:9053/
  curl http://localhost:9053/api
  ```

- [x] Run the oracle check and confirm it prints `1.000`:

  ```bash
  .venv/bin/python local_evaluator.py --oracle
  ```

- [x] Run the baseline once to record the starting score (`0.000` on Helsinki):

  ```bash
  .venv/bin/python local_evaluator.py --verbose
  ```

**Gate:** the server starts cleanly, the oracle scores `1.000`, and the baseline response passes local validation.

## Stage 1: Understand the reference data and protocol

**Goal:** make coordinate handling and frame timing reliable before optimizing detection.

- [x] Inspect all 25 Helsinki annotations and confirm there is one instance of each accepted class.
- [x] Render sample annotations:

  ```bash
  .venv/bin/python visualize.py --frame 0 --show
  .venv/bin/python visualize.py --all
  ```

- [x] Write small checks for `utils.view_bbox_to_global`, clipping, and response validation.
- [x] Confirm that detections from Level 1 and Level 2 are normalized by `original_width` and `original_height`, never by the crop size.
- [x] Confirm that invalid boxes, unknown classes, non-integer camera centers, and exceptions cannot escape `predict()`.
- [x] Record inference latency and response latency separately.

**Gate:** a synthetic detection placed in each resolution level maps back to the expected source pixels, and malformed responses are rejected locally.

## Stage 2: Fit the motion model

**Goal:** predict where an object is in the current full frame after it leaves the camera crop.

- [x] Load object centers and box sizes from `src/helsinki/annotations/`.
- [x] Fit a simple per-object motion model from frame number to center position and box geometry in `src/motion.py`.
  Start with a quadratic model for `y`; use a linear model for `x` unless the data justifies more complexity.
- [x] Measure held-out forecast error (15 tracks, mean center-sum error 21.9 px; worst 87.3 px).
- [x] Keep the model independent of the HTTP layer so it can be unit-tested and replaced.
- [x] Add a fallback for unseen or poorly fitted tracks: constant velocity, then last known position.

The fitted model is connected to live detections as a fallback and as a spatial consistency check for weak template matches.

**Gate:** a held-out reference frame forecast has bounded error and no projected box becomes invalid after clipping.

## Stage 3: Build the state and tracking layer

**Goal:** answer for the whole source frame, not only the current crop.

- [x] Add state keyed by `sequence_id`; never mix tracks across attempts.
- [x] Store class, normalized box, confidence, last-seen frame, and motion information.
- [x] Convert detector output from the current view into global coordinates before adding it to state.
- [x] Update observations for the same class and deduplicate by confidence.
- [x] Project every active track to the current `frame` using locally estimated velocity.
- [x] Decay confidence when a track is not directly observed.
- [x] Drop tracks after they leave the source frame or become too uncertain.
- [x] Deduplicate before creating the response; there is no server-side NMS.
- [x] Reset state when a new `sequence_id` appears.

Implementation started in `src/tracking.py`; integration with `predict()` remains open.

**Gate:** replaying the Helsinki sequence with known observations produces one valid track per object and does not emit duplicate boxes.

## Stage 4: Implement the detector

**Goal:** recognize the 16 classes from tiny crops with limited reference data.

### 4.1 Synthetic training data

- [x] Extract the 16 reference object crops and boxes.
- [x] Generate varied scenes using copy-paste augmentation in `src/synthetic_data.py`.
- [ ] Keep a held-out synthetic split and preserve class balance.
- [ ] Train the smallest detector that meets the latency budget.
- [x] Save/load detector templates outside the request path and warm them once at startup.

The OpenCV-only augmentation pipeline is ready. Torch, Ultralytics, and Torchvision are not installed in the current environment, so detector training and model selection remain pending a deliberate dependency/compute choice.

### 4.2 Inference integration

- [x] Replace the placeholder `detect()` implementation in `example.py` with the OpenCV reference-template detector.
- [x] Convert detector boxes with `view_bbox_to_global`.
- [x] Clip and validate every box before adding it to state.
- [x] Apply a confidence threshold and class deduplication.
- [x] Warm up the model before serving requests.
- [x] Catch inference failures and return an empty valid annotation list for that frame.

### 4.3 Optional fallback

- [ ] Add proposal generation plus embedding similarity only if the detector has weak classes.
- [ ] Compare the fallback against the detector on the same held-out examples before enabling it.

**Gate:** the CPU fallback reaches `0.260` offline mAP and about `277 ms` mean request latency on Helsinki. A neural detector remains an optional upgrade requiring additional dependencies and compute.

## Stage 5: Implement the camera policy

**Goal:** collect useful detail while preserving legal movement and broad coverage.

- [x] Read `allowed_resolution_levels`, bounds, and `maximum_center_delta` from every request.
- [x] Start with Level 0, transition through Level 1, and sweep at Level 2.
- [x] Implement a constraint-aware serpentine sweep.
- [x] Change levels only one step at a time; never request Level 0 to Level 2 directly.
- [x] Clamp centers to the bounds for the requested level and cast them to real integers.
- [x] Log `camera_command_feedback` and adapt after a rejected command.
- [x] Keep sweep direction per `sequence_id`.

**Gate:** a local replay produces no rejected camera commands during a complete sequence.

## Stage 6: Integrate and benchmark locally

**Goal:** establish the best reproducible local version before exposing it to the internet.

- [x] Wire detector, motion model fallback, state layer, and camera policy through `predict()`.
- [x] Keep `api.py` and the response schema unchanged unless a protocol issue is found.
- [x] Run the non-realtime evaluator (`0.260` mAP, 25/25 responses accepted):

  ```bash
  .venv/bin/python local_evaluator.py
  ```

- [x] Run the realtime evaluator (`0.170` mAP, 15/25 frames delivered):

  ```bash
  .venv/bin/python local_evaluator.py --realtime
  .venv/bin/python local_evaluator.py --realtime --simulate-latency-ms 400
  ```

- [x] Compare non-realtime and realtime scores; the gap identifies frame cadence as the current limitation.
- [x] Stress-test with 400 ms simulated latency (`0.110` mAP, 9/25 frames delivered).
- [ ] Run repeated trials and record score, skipped frames, rejected camera commands, and median/p95 latency.
- [x] Add a startup smoke test that imports the model and performs one warm-up inference.
- [ ] Freeze a known-good local configuration and keep a rollback copy.

**Gate:** the endpoint handles a full local replay without crashes, illegal commands, or invalid responses. Frame loss remains the main optimization target before remote submission.

## Stage 7: Optimize realtime latency and frame retention

**Goal:** reduce skipped frames before changing detector behavior further.

- [ ] Add per-request timing for decode, detector, tracker, response construction, and total round trip.
- [ ] Run at least five realtime trials and record mean, p95, maximum latency, sent frames, and skipped frames.
- [ ] Cache grayscale/resized templates and avoid repeated disk access inside `predict()`.
- [ ] Benchmark one detector pass, one scale, and a reduced candidate set separately.
- [ ] Keep detector warm-up outside the request path.
- [ ] Target p95 request latency below `250 ms` and at least `22/25` frames delivered on Helsinki.
- [ ] Re-run the offline score after every latency change to detect accuracy regressions.

**Gate:** the optimized implementation improves realtime frame retention without reducing offline mAP or introducing invalid responses.

The resized-template cache is implemented in `src/template_detector.py`. The first benchmark was noisy and did not improve retention (`16/25` frames, `0.198` mAP), so it is retained for warm-cache behavior but requires repeated trials before being accepted as a score improvement.

## Stage 8: Recover zero-score classes

**Goal:** improve macro mAP by addressing classes currently scoring `0.000` rather than over-optimizing already strong classes.

- [ ] Build a per-class evaluation table with prediction count, matched count, false positives, and mean IoU.
- [ ] Inspect crops and template scores for `hangar`, `jet_plane`, `large_tower`, `medium_launcher`, and `medium_plane`.
- [ ] Extract multiple reference templates per class instead of using only the first annotation.
- [ ] Try edge-normalized and grayscale templates for classes sensitive to brightness or background.
- [ ] Test class-specific thresholds and scale ranges on a held-out subset of Helsinki.
- [ ] Add targeted synthetic examples for weak classes with rotation, blur, contrast, and scale variation.
- [ ] Keep a class-level regression table so improvements for weak classes do not create duplicate false positives elsewhere.

An initial three-template experiment improved offline mAP only from `0.260` to `0.263` but reduced realtime mAP to `0.166` at roughly `417 ms` per request. It was reverted from the live path.

**Gate:** every class has a measured non-zero candidate path or a documented reason why the current detector cannot support it.

## Stage 9: Improve tracking and whole-frame recall

**Goal:** preserve accurate detections across skipped frames and camera moves.

- [ ] Replay known Helsinki observations with artificial frame gaps of 1, 2, and 3 frames.
- [ ] Compare constant-velocity projection against the fitted reference motion model on held-out tracks.
- [ ] Use class-specific confidence decay based on measured detector stability.
- [ ] Expand track matching from class-only replacement to IoU/proximity matching when multiple candidates exist.
- [ ] Reject projected boxes whose motion or confidence becomes implausible.
- [ ] Verify that one object produces at most one response annotation per frame.
- [ ] Measure recall separately for directly observed and carried-forward detections.

**Gate:** tracking improves gapped-replay recall without increasing duplicate predictions or invalid boxes.

## Stage 10: Tune camera coverage

**Goal:** choose views that maximize useful detections while preserving legal movement.

- [ ] Compare Level 0, Level 1, and Level 2 policies with the same detector and tracker.
- [ ] Measure which source regions and classes are actually observed during each sweep.
- [ ] Add a leading-edge bias only after confirming the flight direction from frame data.
- [ ] Use Level 2 for confirmation only when a candidate or uncertain region justifies it.
- [ ] Test alternative sweep step sizes against both offline and realtime scores.
- [ ] Confirm every policy variant produces zero rejected camera commands.

**Gate:** select the policy with the best realtime mAP and frame coverage, not merely the best offline detector score.

## Stage 11: Freeze and regression-test the local release

**Goal:** establish a reproducible version before exposing the service publicly.

- [ ] Run oracle validation and the full non-realtime evaluator.
- [ ] Run at least five normal realtime trials and one 400 ms latency stress trial.
- [ ] Record score, sent/skipped frames, response latency, camera rejections, and per-class AP.
- [ ] Run response-schema, coordinate, tracker-gap, and camera-constraint tests.
- [ ] Freeze the selected templates, thresholds, camera policy, and dependency environment.
- [ ] Save a rollback copy of the complete working directory and model artifacts.

**Gate:** the release has reproducible metrics, no protocol failures, and a known rollback path.

## Stage 12: Prepare port forwarding

**Goal:** make the exact local server reachable from the competition service.

- [ ] Choose the machine that will run the server and give it a reserved/static LAN address.
- [ ] Confirm the server listens on `0.0.0.0:9053`, not only on `localhost`.
- [ ] Configure the router to forward an external TCP port to the machine's port `9053`.
- [ ] Allow the port through the machine firewall if required.
- [ ] Keep the submitted path exactly `/predict`.
- [ ] Disable sleep and automatic shutdown for the evaluation window.
- [ ] Record the public address and verify whether the public IP can change.
- [ ] If the network blocks inbound forwarding, switch early to a teammate's home network or a cloud VM.

**Gate:** from a device on mobile data or another external network, these checks succeed:

```bash
curl http://<public-host>:<external-port>/
curl http://<public-host>:<external-port>/api
```

The submitted endpoint must be:

```text
http://<public-host>:<external-port>/predict
```

## Stage 13: Remote dry run and submission

**Goal:** validate the deployed copy, not merely the code on the development machine.

- [ ] Copy the locked code, model files, and dependency environment to the deployment machine.
- [ ] Start the server with the same command used for the final attempt.
- [ ] Warm up the model before submitting the URL.
- [ ] Run the external verification request.
- [ ] Run a validation attempt and record the score and any protocol/timing failures.
- [ ] Check logs for skipped frames, rejected camera commands, exceptions, and slow requests.
- [ ] Do one final local and remote smoke test immediately before evaluation.
- [ ] Submit the URL with `/predict` included and leave the machine awake and connected.

## Shared response checklist

Before every serious run, verify:

- [ ] `request_id` and `frame` are echoed exactly.
- [ ] Every `object_id` is in `dtos.OBJECT_CLASSES`.
- [ ] Every box is global, normalized, finite, and satisfies `0 <= x1 < x2 <= 1` and `0 <= y1 < y2 <= 1`.
- [ ] No duplicate prediction is emitted for one object.
- [ ] `requested_view` centers are integers and obey the current constraints.
- [ ] Exceptions become valid empty responses rather than missing responses.
- [ ] Model warm-up occurs before the first scored request.
- [ ] Request latency leaves margin below the `3333 ms` timeout and the `333 ms` frame interval.
