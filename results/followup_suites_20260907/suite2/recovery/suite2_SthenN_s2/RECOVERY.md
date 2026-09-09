# Recovery of the interrupted Suite 2 call

The original call stopped before it saved an adapter or a run configuration. Its partial trace contains 3,832 row visits across 958 batches.

The frozen job requires 12,720 row visits across six epochs. The original call cannot supply a completed adapter or an optimizer checkpoint.

We restarted only `suite2_SthenN_s2` from its saved merged parent, `suite2_S_s2`. The new attempt uses the frozen dataset, seed, recipe, and training function.

The recovery wrapper uses a separate Modal claim. It preserves the original claim, original handles, original failed outcome, and original directory bytes.

The original remote directory now lives under `retrain_v1/original_attempt`. Its file hashes must match the read-only inspection before recovery can start.

The collector preserves the original local STARTED record under the same archive path within `raw`. It records the authorized change in `LOCAL_ATTEMPT_TRANSITION.json`.

The recovered call writes new canonical training artifacts. The CPU audit will verify those artifacts and all other planned training tags.

The recovery also dispatches the three missing evaluation jobs. It uses the frozen generation function, batteries, sampling settings, and response seals.

The original suite outcome remains unchanged. A separate collection receipt will link that outcome to the verified recovery outcome and complete response count.

The experiment has 28 planned training tags and 29 training attempts, including the interrupted attempt. The partial attempt does not contribute weights to the final adapter.

The saved trace records forward calls. A preemption can interrupt the final recorded forward call, so the partial trace does not prove completed optimizer updates.
