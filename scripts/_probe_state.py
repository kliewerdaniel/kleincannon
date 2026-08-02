import os, json, time, urllib.request, subprocess

ep = "/Users/danielkliewer/Documents/Projects/kleincannon/episodes/2026-08-01-why-the-ocean-is-salty"

def mt(p):
    return os.path.getmtime(p) if os.path.exists(p) else None

files = {
    "voice.wav": f"{ep}/audio/voice.wav",
    "episode.json": f"{ep}/episode.json",
    "captions/words.json": f"{ep}/captions/words.json",
    "captions/frames dir": f"{ep}/captions/frames",
}
print("=== mtimes (sorted newest first) ===")
items = [(k, mt(v)) for k, v in files.items()]
for k, t in sorted(items, key=lambda x: -(x[1] or 0)):
    print(f"  {time.strftime('%H:%M:%S %m-%d', time.localtime(t)) if t else 'MISSING':22}  {k}")

# comfy up?
try:
    urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3)
    comfy = "UP"
except Exception as e:
    comfy = f"DOWN ({e})"
# llm gemma up?
try:
    urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3)
    llm = "UP"
except Exception as e:
    llm = f"DOWN ({e})"

print("ComfyUI:", comfy)
print("LLM gemma:", llm)

# count caption frames
fd = f"{ep}/captions/frames"
if os.path.isdir(fd):
    n = len([f for f in os.listdir(fd) if f.startswith("frame_")])
    print("caption frames:", n)
