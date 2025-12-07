# run_scas.py
from pathlib import Path
import subprocess
import sys
import webbrowser


def main():
    root = Path(__file__).resolve().parent
    app_path = root / "ui" / "streamlit_app.py"

    port = "8501"
    url = f"http://127.0.0.1:{port}"

    # Start Streamlit using the same Python
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address", "127.0.0.1",
            "--server.port", port,
        ]
    )

    # Open browser
    webbrowser.open(url)

    # Wait for streamlit server to exit
    proc.wait()


if __name__ == "__main__":
    main()
