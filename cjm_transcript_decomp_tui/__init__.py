import os

__version__ = "0.0.1"

# PipeWire routing (correction-TUI craft 7d7e37af): conda-forge PortAudio ships
# an ALSA that cannot see the system pipewire/default PCM (hw-only enumeration
# -> audio lands on the wrong sink). Point it at the system ALSA config HERE,
# in the package __init__, because canonical emit hoists a module's own import
# block above any module-level code — only the package boundary runs strictly
# before a submodule imports sounddevice. Pre-set env always wins (setdefault).
if os.path.exists("/usr/share/alsa/alsa.conf"):
    os.environ.setdefault("ALSA_CONFIG_PATH", "/usr/share/alsa/alsa.conf")
if os.path.isdir("/usr/lib/x86_64-linux-gnu/alsa-lib"):
    os.environ.setdefault("ALSA_PLUGIN_DIR", "/usr/lib/x86_64-linux-gnu/alsa-lib")
