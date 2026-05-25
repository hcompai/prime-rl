import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import verifiers as vf
import wandb
from transformers.tokenization_utils import PreTrainedTokenizer
from wandb.errors import CommError
from wandb.sdk.mailbox.mailbox_handle import ServerResponseError

from prime_rl.configs.shared import WandbConfig, WandbWithExtrasConfig
from prime_rl.utils.chat_template import deserialize_tool_calls
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor, sample_items_for_logging


class WandbMonitor(Monitor):
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbConfig | WandbWithExtrasConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseConfig | None = None,
        keep_full_history: bool = True,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self._keep_full_history = keep_full_history
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0

        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")
        self._maybe_overwrite_wandb_command()

        # PATCH: WANDB_MODE=disabled/offline must override shared mode (shared mode requires a server connection).
        shared_mode = os.environ.get("WANDB_SHARED_MODE") == "1" and os.environ.get("WANDB_MODE") not in (
            "disabled",
            "offline",
        )
        if shared_mode:
            run_id = os.environ.get("WANDB_SHARED_RUN_ID")
            label = os.environ.get("WANDB_SHARED_LABEL")
            primary = label == "orchestrator"
            settings = wandb.Settings(
                mode="shared",
                x_label=label,
                x_primary=primary,
                x_update_finish_state=primary,
            )
            self.logger.info(
                f"Using shared W&B mode ({label=}, {primary=}). "
                "This is an experimental feature. Disable with --wandb.shared False"
            )
        else:
            run_id = None
            primary = False
            settings = wandb.Settings(
                mode=os.environ.get("WANDB_MODE", "offline" if config.offline else "online"),  # PATCH: honor WANDB_MODE
            )

        # PATCH: also retry on ServerResponseError in shared mode (transient on upsertBucket).
        retryable_errors = (CommError, ServerResponseError) if shared_mode else (CommError,)

        def init_wandb(max_retries: int):
            for attempt in range(max_retries):
                try:
                    return wandb.init(
                        id=run_id,
                        resume="allow" if run_id else None,  # PATCH: allow re-attach when run_id is set
                        project=config.project,
                        entity=config.entity,
                        name=config.name,
                        group=config.group,
                        tags=config.tags,
                        dir=output_dir,
                        config=run_config.model_dump() if run_config else None,
                        settings=settings,
                    )
                except retryable_errors as e:
                    if attempt + 1 == max_retries:
                        raise
                    if shared_mode and not primary:
                        msg = (
                            f"Shared W&B run not yet created by primary - retrying in 10s ({attempt + 1}/{max_retries})"
                        )
                    else:
                        msg = f"Transient W&B init error ({e}) - retrying in 10s ({attempt + 1}/{max_retries})"
                    self.logger.info(msg)
                    # PATCH: failed wandb.init leaves run_id registered in wandb-core StreamMux, causing the next
                    # attempt to fail with "run ID ... is in use". Tear down so the retry starts clean.
                    wandb.teardown()
                    time.sleep(10)

        # Non-primary processes in shared mode wait for the primary to create the run.
        # Everyone else still retries to absorb transient W&B server errors (e.g. 404 on upsertBucket).
        max_retries = 30 if shared_mode and not primary else 5
        self.wandb = init_wandb(max_retries)

        wandb.define_metric("*", step_metric="step")

        # Optionally, initialize sample logging attributes
        if config is not None and isinstance(config, WandbWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples_cols = ["step", "env_name", "task", "example_id", "messages", "input_ids", "reward"]
                self.samples_table = wandb.Table(
                    columns=self.samples_cols,
                    log_mode="INCREMENTAL",
                )
                self.tokenizer = tokenizer
                self.eval_samples_cols = ["step", "env", "task", "example_id", "completion", "reward"]
                self.eval_samples_table = wandb.Table(
                    columns=self.eval_samples_cols,
                    log_mode="INCREMENTAL",
                )
                # PATCH: dedicated traces tables — one dense row per log interval, easier to find than
                # a sparse `trace_html` column inside the much larger samples table. `trace_html` is
                # placed first so W&B's table viewer shows it by default (its auto-hide heuristic
                # hides large media columns when they're not at the start of the column list).
                self.traces_cols = ["trace_html", "step", "env_name", "task", "example_id", "reward"]
                self.traces_table = wandb.Table(columns=self.traces_cols, log_mode="INCREMENTAL")
                self.eval_traces_cols = ["trace_html", "step", "env", "task", "example_id", "reward"]
                self.eval_traces_table = wandb.Table(columns=self.eval_traces_cols, log_mode="INCREMENTAL")
                # PATCH: pin the `trace_html` column to `wandb.Html` up front so a failed first render
                # (None from `_safe_render_html`) can't lock the column to string and stop the UI from
                # rendering subsequent cells as HTML. `optional=True` keeps None values legal.
                self.traces_table.cast("trace_html", wandb.Html, optional=True)
                self.eval_traces_table.cast("trace_html", wandb.Html, optional=True)

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._keep_full_history:
            self.history.append(metrics)
        else:
            self.history = [metrics]
        if not self.is_master:
            return
        if not self.enabled:
            return
        wandb.log({**metrics, "step": step})

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        """Logs rollouts to W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log samples if not enabled or not log interval step
            return

        rollouts = sample_items_for_logging(
            rollouts,
            self.config.log_extras.sample_ratio,
        )
        if not rollouts:
            return

        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging {len(rollouts)} samples to W&B table at step {step}")
        start_time = time.perf_counter()

        # PATCH: render exactly one rollout to the dedicated traces table — keeps the table dense
        # (one row per interval) instead of leaking sparse trace_html cells into the samples table.
        # A 30-turn OSWorld trace is ~1-2 MB; logging every rollout would be hundreds of MB per run.
        trace_rollout = random.choice([r for r in rollouts if r.get("trajectory")] or [None])

        for rollout in rollouts:
            trajectory = rollout["trajectory"]
            if not trajectory:
                continue
            last_step = trajectory[-1]
            tokens = last_step["tokens"]
            full_ids = tokens["prompt_ids"] + tokens["completion_ids"]
            messages_text = self.tokenizer.decode(full_ids)
            sample = {
                "step": step,
                "env_name": rollout.get("env_name"),
                "task": rollout.get("task"),
                "example_id": rollout["example_id"],
                "messages": messages_text,
                "input_ids": str(full_ids),
                "reward": rollout["reward"],
            }
            assert list(sample.keys()) == self.samples_cols, (
                "Order of columns in the table must be the same as order of the keys here"
            )
            self.samples_table.add_data(*sample.values())

        # PATCH: append the picked trace to the dedicated traces table — at most once per
        # `trace_interval` steps (defaults to 5x the samples interval) so the run artifacts don't
        # balloon. Override per-config with [orchestrator.wandb.log_extras] trace_interval = N.
        trace_interval = getattr(self.config.log_extras, "trace_interval", None) or self.config.log_extras.interval * 5
        if trace_rollout is not None and step % trace_interval == 0:
            self.traces_table.add_data(
                _safe_render_html(trace_rollout, self.logger),
                step,
                trace_rollout.get("env_name"),
                trace_rollout.get("task"),
                trace_rollout["example_id"],
                trace_rollout["reward"],
            )
            wandb.log({"traces": self.traces_table, "step": step})

        wandb.log({"samples": self.samples_table, "step": step})
        self.last_log_samples_step = step
        self.logger.debug(f"Logged samples at step {step} to W&B table in {time.perf_counter() - start_time:.2f}s")

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        """Logs eval rollouts to a separate W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
        ):
            return

        # PATCH: render exactly one eval rollout to the dedicated eval traces table (see log_samples).
        trace_rollout = random.choice([r for r in rollouts if r.get("completion")] or [None])

        for rollout in rollouts:
            completion = rollout.get("completion")
            if not completion:
                continue
            if isinstance(completion, list):
                try:
                    completion = self.tokenizer.apply_chat_template(deserialize_tool_calls(completion), tokenize=False)
                except Exception:
                    completion = str(completion)
            sample = {
                "step": step,
                "env": env_name,
                "task": rollout.get("task"),
                "example_id": rollout["example_id"],
                "completion": completion,
                "reward": rollout["reward"],
            }
            self.eval_samples_table.add_data(*sample.values())

        # PATCH: append the picked eval trace to the dedicated eval/traces table (see log_samples
        # for the rationale on trace_interval).
        trace_interval = getattr(self.config.log_extras, "trace_interval", None) or self.config.log_extras.interval * 5
        if trace_rollout is not None and step % trace_interval == 0:
            self.eval_traces_table.add_data(
                _safe_render_html(trace_rollout, self.logger),
                step,
                env_name,
                trace_rollout.get("task"),
                trace_rollout["example_id"],
                trace_rollout["reward"],
            )
            wandb.log({"eval/traces": self.eval_traces_table, "step": step})

        wandb.log({"eval/samples": self.eval_samples_table, "step": step})

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """Log distributions (no-op for W&B)."""
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to W&B table."""
        if not self.is_master or not self.enabled:
            return

        self.logger.info("Saving final summary to file")
        assert self.output_dir is not None, "Output directory is required for saving final summary"
        dir_path = self.output_dir / f"run-{self.wandb.id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / filename, "w") as f:
            json.dump(wandb.summary._as_dict(), f)


# PATCH: resolve a renderer lazily from PRIME_RL_TRACE_RENDERER="module.path:func". Returning None
# disables HTML rendering (the trace_html column stays empty) without crashing the monitor.
_TRACE_RENDERER_CACHE: tuple[str | None, Callable[[vf.RolloutOutput], str] | None] = (None, None)


def _get_trace_renderer(logger: Any) -> Callable[[vf.RolloutOutput], str] | None:
    global _TRACE_RENDERER_CACHE
    spec = os.environ.get("PRIME_RL_TRACE_RENDERER") or None
    if _TRACE_RENDERER_CACHE[0] == spec:
        return _TRACE_RENDERER_CACHE[1]
    fn: Callable[[vf.RolloutOutput], str] | None = None
    if spec:
        try:
            module_path, _, attr = spec.partition(":")
            fn = getattr(importlib.import_module(module_path), attr)
        except Exception as e:
            logger.warning(f"PRIME_RL_TRACE_RENDERER={spec!r} could not be resolved: {e}")
    _TRACE_RENDERER_CACHE = (spec, fn)
    return fn


# PATCH: render one rollout to wandb.Html, swallowing render errors so the trace column never breaks the table.
def _safe_render_html(rollout: vf.RolloutOutput, logger: Any) -> wandb.Html | None:
    renderer = _get_trace_renderer(logger)
    if renderer is None:
        return None
    try:
        return wandb.Html(renderer(rollout))
    except Exception as e:
        logger.warning(f"Trace HTML render failed for example_id={rollout.get('example_id')}: {e}")
        return None
