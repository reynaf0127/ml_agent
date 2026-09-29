import os
import sys
from datetime import datetime


class TeeLogger:
    """
    Write stdout to both:
        1. terminal
        2. Markdown run report
    """

    def __init__(
        self,
        terminal,
        file_handle,
    ):
        self.terminal = terminal
        self.file_handle = file_handle

    def write(self, message):

        self.terminal.write(
            message
        )

        self.file_handle.write(
            message
        )

    def flush(self):

        self.terminal.flush()
        self.file_handle.flush()


class RunLogger:

    def __init__(
        self,
        model_name,
        target_col,
        base_dir="artifacts/runs",
    ):

        self.model_name = model_name
        self.target_col = target_col

        # ----------------------------------------------------
        # Run folder
        # ----------------------------------------------------

        run_name = (
            f"{model_name}_{target_col}"
        )

        self.run_dir = os.path.join(
            base_dir,
            run_name,
        )

        self.model_dir = os.path.join(
            self.run_dir,
            "model",
        )

        self.evaluation_dir = os.path.join(
            self.run_dir,
            "evaluation",
        )

        self.interpretation_dir = os.path.join(
            self.run_dir,
            "interpretation",
        )

        # ----------------------------------------------------
        # Create folders
        # ----------------------------------------------------

        for directory in [
            self.run_dir,
            self.model_dir,
            self.evaluation_dir,
            self.interpretation_dir,
        ]:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        # ----------------------------------------------------
        # Markdown report
        # ----------------------------------------------------

        self.report_path = os.path.join(
            self.run_dir,
            "run_report.md",
        )

        self.file = None
        self.original_stdout = None

    # ========================================================
    # START LOGGING
    # ========================================================

    def start(self):

        self.file = open(
            self.report_path,
            "w",
            encoding="utf-8",
        )

        # Markdown header
        self.file.write(
            f"# ML Model Run — "
            f"{self.model_name}\n\n"
        )

        self.file.write(
            f"**Target:** `{self.target_col}`\n\n"
        )

        self.file.write(
            f"**Model:** `{self.model_name}`\n\n"
        )

        self.file.write(
            f"**Run time:** "
            f"{datetime.now().isoformat(timespec='seconds')}\n\n"
        )

        self.file.write(
            "## Terminal Output\n\n"
        )

        self.file.write(
            "```text\n"
        )

        self.file.flush()

        # Save original stdout
        self.original_stdout = (
            sys.stdout
        )

        # Replace stdout
        sys.stdout = TeeLogger(
            terminal=self.original_stdout,
            file_handle=self.file,
        )

        print(
            "\n[RUN] Logging started."
        )

        print(
            f"[RUN] Directory: "
            f"{self.run_dir}"
        )

    # ========================================================
    # STOP LOGGING
    # ========================================================

    def stop(self):

        if self.file is None:
            return

        print(
            "\n[RUN] Logging complete."
        )

        # Restore terminal
        sys.stdout = (
            self.original_stdout
        )

        self.file.write(
            "\n```\n"
        )

        self.file.close()

        print(
            f"\n[RUN] Report saved:"
            f"\n{self.report_path}"
        )