#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define LDR_PIN 34
#define OLED_SDA 21
#define OLED_SCL 22
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

constexpr int kTensorArenaSize = 12 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const char* labels[] = {"Dark", "Normal", "Bright"};

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  Wire.begin(OLED_SDA, OLED_SCL);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED initialization failed");
    while (true) delay(1000);
  }

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema mismatch");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddQuantize();

  static tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (true) delay(1000);
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  while (true) {
    const int raw = analogRead(LDR_PIN);
    const float normalized = raw / 4095.0f;

    input->data.f[0] = normalized;

    if (interpreter.Invoke() != kTfLiteOk) {
      Serial.println("Inference failed");
      delay(500);
      continue;
    }

    int best = 0;
    for (int i = 1; i < 3; ++i) {
      if (output->data.f[i] > output->data.f[best]) {
        best = i;
      }
    }

    Serial.printf("LDR=%d  input=%.3f  class=%s\n",
                  raw, normalized, labels[best]);

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Live Sensor + TinyML");

    display.setTextSize(2);
    display.setCursor(0, 18);
    display.println(labels[best]);

    display.setTextSize(1);
    display.setCursor(0, 45);
    display.print("LDR: ");
    display.println(raw);
    display.setCursor(0, 55);
    display.print("Input: ");
    display.println(normalized, 2);
    display.display();

    delay(500);
  }
}
