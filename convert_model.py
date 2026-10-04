from pathlib import Path

MODEL = Path("light_level_classifier.tflite")
OUTPUT = Path("model.h")

if not MODEL.exists():
    raise SystemExit(
        "light_level_classifier.tflite not found. Run Live_Sensor_TinyML.ipynb first."
    )

data = MODEL.read_bytes()

with OUTPUT.open("w", encoding="utf-8") as f:
    f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write("alignas(8) const unsigned char g_model[] = {\n")
    for i in range(0, len(data), 12):
        f.write("  " + ", ".join(f"0x{b:02x}" for b in data[i:i+12]) + ",\n")
    f.write("};\n")
    f.write("const unsigned int g_model_len = sizeof(g_model);\n\n")
    f.write("#endif\n")

print(f"Generated {OUTPUT} from {len(data)} model bytes.")
