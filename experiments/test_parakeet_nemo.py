import sys
from pathlib import Path

# Remove BridgeDEUX project root so "datasets" resolves to
# Hugging Face datasets instead of D:\BridgeDEUX\datasets.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path = [
    p for p in sys.path
    if Path(p or ".").resolve() != PROJECT_ROOT
]

import nemo.collections.asr as nemo_asr


MODEL = r"D:\BridgeDEUX\models\parakeet\nemo\parakeet-tdt-0.6b-v3.nemo"
AUDIO = r"D:\BridgeDEUX\datasets\raw\mslt\de_en\test\MSLT_Test_DE_0001.T0.de.wav"


print("Loading NVIDIA Parakeet-TDT 0.6B v3...")

model = nemo_asr.models.ASRModel.restore_from(
    restore_path=MODEL,
    map_location="cpu",
)

print("Model loaded.")
print("Transcribing:", AUDIO)

output = model.transcribe(
    [AUDIO],
    batch_size=1,
)

print("\nTEXT:")
for item in output:
    print(item)