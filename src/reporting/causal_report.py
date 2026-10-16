import sys
from pathlib import Path
from datetime import datetime


class TeeOutput:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class CausalMarkdownReport:

    def __init__(self, output_dir="artifacts/causal"):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        terminal_output,
        filename="causal_analysis_report.md",
    ):

        path = self.output_dir / filename

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        markdown = (
            "# Causal Analysis Report\n\n"
            f"Generated: {timestamp}\n\n"
            "---\n\n"
            "## Workflow Output\n\n"
            f"{terminal_output}\n"
        )

        path.write_text(
            markdown,
            encoding="utf-8",
        )

        print(
            "\n[REPORT] Saved causal analysis report:"
            f"\n{path}"
        )

        return str(path)