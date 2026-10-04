# Live Sensor + TinyML

ESP32 project that reads a live LDR value, runs a small TensorFlow Lite Micro model on-device, and shows **Dark, Normal, or Bright** on an SSD1306 OLED.

## Connections
- LDR analog output -> GPIO 34
- OLED SDA -> GPIO 21
- OLED SCL -> GPIO 22
- OLED VCC -> 3.3V
- OLED GND -> GND

## TinyML classes
- 0 = Dark
- 1 = Normal
- 2 = Bright

The LDR reading is normalized from 0-4095 to 0.0-1.0 before inference.

## Files
- `sketch.ino` - ESP32 live sensor + TFLite Micro inference + OLED
- `Live_Sensor_TinyML.ipynb` - trains and exports the TFLite model
- `convert_model.py` - converts the .tflite file to `model.h`
- `model.h` - generated model header placeholder; replace it after running the conversion script
- `diagram.json` - Wokwi circuit
- `libraries.txt` - required Arduino libraries

## Model setup
Run the notebook in Google Colab. It creates `light_level_classifier.tflite`. Then run:

```bash
python convert_model.py
```

Copy the generated `model.h` into this repository and compile the ESP32 sketch with an ESP32-compatible TensorFlow Lite Micro library.

The ESP32 performs the classification locally; no cloud service is used.
