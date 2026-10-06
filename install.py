import importlib.util
import shutil
import subprocess
import sys

is_colab = "google.colab" in sys.modules
is_kaggle = "kaggle_secrets" in sys.modules


def _run(cmd, error_msg):
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        print(process.stderr[-2000:])
        raise Exception(error_msg)
    return process


def _pip_install(args, error_msg):
    """Install packages into the running kernel's environment.

    Environments created with `uv venv` don't include pip, so fall back to
    `uv pip` (or bootstrap pip with ensurepip) instead of failing.
    """
    if importlib.util.find_spec("pip") is None:
        if shutil.which("uv"):
            return _run(["uv", "pip", "install", "--python", sys.executable, *args], error_msg)
        _run([sys.executable, "-m", "ensurepip", "--upgrade"], "😭 pip is missing and could not be installed")
    return _run([sys.executable, "-m", "pip", "install", "-q", *args], error_msg)


def install_requirements(
    is_chapter10: bool = False,
    is_chapter11: bool = False,
    ):
    """Installs the required packages for the project.

    Only needed on hosted platforms like Colab or Kaggle. Locally, create the
    environment once with `uv pip install -r requirements.txt` (see README.md).
    """

    print("⏳ Installing base requirements ...")
    _pip_install(["-r", "requirements.txt"], "😭 Failed to install base requirements")
    print("✅ Base requirements installed!")

    if is_chapter10:
        print("⏳ Installing wandb ...")
        _pip_install(["wandb"], "😭 Failed to install wandb")
        print("✅ wandb installed!")

    if is_chapter11 and (is_colab or is_kaggle):
        # torchcodec (audio decoding in datasets) needs the FFmpeg libraries
        print("⏳ Installing soundfile and ffmpeg ...")
        subprocess.run(["apt", "install", "-y", "libsndfile1", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ soundfile and ffmpeg installed!")

    print("🥳 Chapter installation complete!")
