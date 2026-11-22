"""Launch the script LLM exactly as the pipeline would.

watchdog.bat used to hardcode its own copy of the model path, the exe and the
server flags. It drifted: it named a model the pipeline had moved on from, used
a llama-server.exe that config.json does not use, and omitted --reasoning off.
Reading the same source of truth removes the class of bug, not the instance.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_producer as ep  # noqa: E402


def main():
    vision = "--vision" in sys.argv
    cmd = ep._llm_launch_cmd(vision=vision)
    if not Path(cmd[0]).is_file():
        print(f"llama-server not found: {cmd[0]}", file=sys.stderr)
        return 2
    subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print("launched " + " ".join(str(c) for c in cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
