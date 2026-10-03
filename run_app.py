"""Launch the Streamlit app and reset the terminal when it exits."""

import os
import subprocess
import sys


def main() -> None:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_dir, "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]

    try:
        subprocess.run(cmd, cwd=project_dir, check=False)
    finally:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        print("App closed. Terminal reset.")
        print("Start again with: python run_app.py")


if __name__ == "__main__":
    main()
