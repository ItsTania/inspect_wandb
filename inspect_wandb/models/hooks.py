import logging
from pathlib import Path
from typing import Any
from typing_extensions import override

import wandb
from inspect_wandb.utils.ansi import strip_ansi_from_file
from wandb.errors import CommError
from inspect_ai.hooks import Hooks, RunEnd, SampleEnd, TaskStart, EvalSetStart
from inspect_ai.log import EvalSample
from inspect_ai.scorer import CORRECT
from inspect_ai.util._display import display_type
from inspect_wandb.config.settings import ModelsSettings
from inspect_wandb.config.extras_manager import INSTALLED_EXTRAS
if INSTALLED_EXTRAS["viz"]:
    from inspect_wandb.viz.inspect_viz_writer import InspectVizWriter

logger = logging.getLogger(__name__)

class Metric:
    ACCURACY: str = "accuracy"
    SAMPLES: str = "samples"

class WandBModelHooks(Hooks):

    settings: ModelsSettings | None = None

    _correct_samples: int = 0
    _total_samples: int = 0
    _wandb_initialized: bool = False
    _is_eval_set: bool = False
    _hooks_enabled: bool | None = None
    _active_runs: dict[str, dict[str, bool | BaseException | None]] = {}
    _metadata_overrides: dict[str, Any] | None = None
    _plain_log_path: str | None = None

    def __init__(self):
        if INSTALLED_EXTRAS["viz"]:
            self.viz_writer = InspectVizWriter()
        else:
            self.viz_writer = None

    @override
    def enabled(self) -> bool:
        # Always reload settings from scratch to pick up any runtime changes
        self.settings = ModelsSettings.model_validate(self._metadata_overrides or {})
        return self.settings.enabled

    @override
    async def on_eval_set_start(self, data: EvalSetStart) -> None:
        """
        Hook to run at the start of an eval set.
        Sets a flag to indicate that this is an eval set run, and adds log_dir to state
        """
        self._is_eval_set = True
        self.eval_set_log_dir = data.log_dir
    
    @override
    async def on_run_end(self, data: RunEnd) -> None:
        # Only proceed with cleanup if WandB was actually initialized
        if not self._wandb_initialized:
            return

        self._active_runs[data.run_id]["running"] = False
        if data.exception is not None:
            self._active_runs[data.run_id]["exception"] = data.exception

        self._log_summary(data)

        # Upload the terminal log file to wandb at end of run.
        if display_type() == "full_log":
            try:
                from inspect_ai._display.full_log.display import default_plain_log_path
                from pathlib import Path as _Path
                log_path = self._plain_log_path or default_plain_log_path()
                if _Path(log_path).exists():
                    self.run.save(log_path, policy="now")
                    logger.info(f"Uploaded terminal log to wandb: {log_path}")
                else:
                    logger.warning(f"Terminal log not found, skipping upload: {log_path}")
            except Exception as e:
                logger.warning(f"Could not upload terminal log: {e}")
            self._plain_log_path = None

        if self.settings is not None and self.settings.viz and self.viz_writer is not None:
            await self.viz_writer.log_scores_heatmap(data, self.run)

        if self.settings is not None and self.settings.files:
            for file in self.settings.files:
                file_path = Path(file)
                if file_path.exists():
                    try:
                        self.run.save(str(file), policy="now")  # TODO: fix wandb Symlinked warning for folder upload
                        logger.info(f"Successfully saved {file} to wandb")
                    except Exception as e:
                        logger.warning(f"Failed to save {file} to wandb: {e}")
                else:
                    logger.warning(f"File or folder '{file}' does not exist. Skipping wandb upload.")

        # Strip ANSI codes from output.log for cleaner wandb logs
        if self.settings is not None and self.settings.strip_ansi_from_output:
            self._strip_ansi_from_output_log()

        if data.exception is not None and isinstance(data.exception, KeyboardInterrupt):
            logger.error("Inspect exited due to KeyboardInterrupt")
            self.run.finish(exit_code=1)
        elif data.exception is not None and isinstance(data.exception, SystemExit):
            logger.error(f"SystemExit running eval set: {data.exception}")
            self.run.finish(exit_code=3)
        elif (last_run:= all([not run["running"] for run in self._active_runs.values()])) and data.exception is not None:
            logger.error("Inspect exited due to exception")
            self.run.finish(exit_code=2)
        elif not(all(log.status == "success" for log in data.logs)) and last_run:
            logger.warning("One or more tasks failed, may retry if eval-set")
            self.run.finish(exit_code=4)
        elif last_run:
            self.run.finish(exit_code=0)

        self._wandb_initialized = False

    @override
    async def on_task_start(self, data: TaskStart) -> None:
        """
        Hook to run at the start of each inspect task.
        Initializes WandB run if not already initialized.
        Updates tags, config, and other metadata based on user-provided settings.
        """
        assert self.settings is not None
        
        # Override settings from eval metadata on first task only
        if self._hooks_enabled is None:
            self._metadata_overrides = self._extract_settings_overrides_from_eval_metadata(data)
            self.settings = ModelsSettings.model_validate(self._metadata_overrides or {})
            self._hooks_enabled = self.settings.enabled
        
        if not self._hooks_enabled:
            logger.info(f"WandB model hooks disabled for run (task: {data.spec.task})")
            return
        
        if self._is_eval_set:
            wandb_run_id = data.eval_set_id
        else:
            wandb_run_id = data.run_id

        self._active_runs[data.run_id] = {
            "running": True,
            "exception": None,
        }

        # Lazy initialization: only init WandB when first task starts
        if not self._wandb_initialized:
            # Determine console capture mode for wandb.
            # When inspect runs with display='full_log' it writes plain text to a
            # log file, so we default console capture to 'off' to avoid interference.
            # Users can override by setting console explicitly in ModelsSettings.
            _tui_displays = {"full_log"}
            if self.settings.console is not None:
                resolved_console = self.settings.console
            elif display_type() in _tui_displays:
                resolved_console = "off"
            else:
                resolved_console = "auto"
            wandb_settings = wandb.Settings(console=resolved_console)

            try:
                self.run = wandb.init(
                    id=wandb_run_id,
                    name=f"Inspect eval-set: {self.eval_set_log_dir}" if self._is_eval_set else None,
                    entity=self.settings.entity,
                    project=self.settings.project,
                    resume="allow",
                    settings=wandb_settings,
                )
            except CommError as e:
                if f"entity {self.settings.entity} not found" in str(e):
                    logger.warning(f"WandB integration disabled: invalid entity: {self.settings.entity}. {e}")
                elif f"project {self.settings.project} not found" in str(e):
                    logger.warning(f"WandB integration disabled: invalid project: {self.settings.project}. {e}")
                else:
                    logger.warning(f"WandB integration disabled: {e}")
                self.settings.enabled = False
                self._hooks_enabled = False
                return

            if self.run.summary:
                self._total_samples = int(self.run.summary.get("samples_total", 0))
                self._correct_samples = int(self.run.summary.get("samples_correct", 0))

            if self.settings.add_metadata_to_config and data.spec.metadata is not None:
                task_id = data.spec.task_id
                metadata_for_wandb = {task_id: {k: v for k,v in data.spec.metadata.items() if k != "inspect_wandb_models_config"}}
                current_wandb_task_metadata: dict[str, Any] = self.run.config.get("inspect task metadata", {})
                wandb_task_metadata = current_wandb_task_metadata | metadata_for_wandb
                self.run.config.update({"inspect task metadata": wandb_task_metadata}, allow_val_change=True)
            if self.settings.config:
                self.run.config.update(self.settings.config, allow_val_change=True)

            _ = self.run.define_metric(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)
            self._wandb_initialized = True
            logger.info(f"WandB initialized for task {data.spec.task}")

            # If the user chose display='full_log', record the plain log path so
            # it can be uploaded to wandb at the end of the run.
            if display_type() == "full_log":
                try:
                    from inspect_ai._display.full_log.display import default_plain_log_path
                    self._plain_log_path = default_plain_log_path()
                    logger.info(f"Plain log will be uploaded to wandb at run end: {self._plain_log_path}")
                except Exception as e:
                    logger.warning(f"Could not resolve plain log path: {e}")
        
            inspect_tags = (
                f"inspect_task:{data.spec.task}"[:64],
                f"inspect_model:{data.spec.model}"[:64],
                f"inspect_dataset:{data.spec.dataset.name}"[:64],
            )
            if self.run.tags:
                self.run.tags = self.run.tags + inspect_tags
            else:
                self.run.tags = inspect_tags

            if self.settings.tags is not None and self.run.tags is not None:
                self.run.tags = self.run.tags + tuple(self.settings.tags)

            data.spec.metadata = (data.spec.metadata or {}) | {"wandb_run_url": self.run.url}

    @override
    async def on_sample_end(self, data: SampleEnd) -> None:
        # Skip if hooks are disabled for this run
        if not self._hooks_enabled:
            return
            
        self._total_samples += 1
        if data.sample.scores:
            self._correct_samples += int(self._is_correct(data.sample))
            self.run.log(
                {Metric.SAMPLES: self._total_samples, Metric.ACCURACY: self._accuracy()}
            )

    def _log_summary(self, data: RunEnd) -> None:
        summary = {
            "samples_total": self._total_samples,
            "samples_correct": self._correct_samples,
            "accuracy": self._accuracy(),
            "logs": [log.location for log in data.logs],
        }
        self.run.summary.update(summary)
        logger.info(f"WandB Summary: {summary}")

    def _is_correct(self, sample: EvalSample) -> bool:
        if not sample.scores:
            return False

        values = [score.value for score in sample.scores.values()]
        return CORRECT in values or 1 in values or 1.0 in values or True in values

    def _accuracy(self) -> float:
        if self._total_samples == 0:
            return 0.0

        return self._correct_samples * 1.0 / self._total_samples

    def _extract_settings_overrides_from_eval_metadata(self, data: TaskStart) -> dict[str, Any] | None:
        """
        Check TaskStart metadata to determine if hooks should be enabled
        """
        if data.spec.metadata is None:
            return None
        overrides = {k[len("inspect_wandb_models_"):]: v for k,v in data.spec.metadata.items() if k.lower().startswith("inspect_wandb_models_")}
        return overrides

    def _strip_ansi_from_output_log(self) -> None:
        """
        Strip ANSI escape codes from wandb's output.log file.

        This makes the Logs tab in wandb readable when running Inspect AI
        with rich/full display mode, by removing terminal formatting codes.
        """
        try:
            output_log_path = Path(self.run.dir) / "output.log"
            if output_log_path.exists():
                strip_ansi_from_file(str(output_log_path))
                logger.debug("Stripped ANSI codes from output.log")
        except Exception as e:
            logger.warning(f"Failed to strip ANSI codes from output.log: {e}")
