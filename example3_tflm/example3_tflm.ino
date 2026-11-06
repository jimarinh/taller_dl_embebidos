#include "esp_camera.h"

// ===========================
// Seleccione el modelo de la camera en board_config.h
// ===========================
#include "board_config.h"


//Configura Tensorflow Lite Micro (TFLM)
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mintomatonet_pruned_hrank0_5_int8.h"  // Modelo TFLite convertido a C

// Objetos TFLM
namespace {
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  int inference_count = 0;

  // Arena de tensores (265KB: Valor máximo)
  constexpr int kTensorArenaSize = 175 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  //Configura la cámara

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  //config.frame_size = FRAMESIZE_UXGA;
  config.frame_size = FRAMESIZE_128X128;
  //config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_DRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;
#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

  //Configura el clasificador
  Serial.println("Inicializando TFLite Micro (modelo int8)...");

  // Carga el modelo
  model = tflite::GetModel(mintomatonet_pruned_hrank0_5_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Versión de modelo incompatible.");
    while (true) {}
  }

  // Registra las operaciones necesarias (según el modelo)
  static tflite::MicroMutableOpResolver<20> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddSoftmax();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddStridedSlice();
  resolver.AddPack();

  // Crea el intérprete
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Asigna tensores
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: No se pudo asignar los tensores.");
    while (true) {}
  }

  // Obtiene tensores de entrada y salida
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Modelo cargado.");
  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(i < input->dims->size - 1 ? " x " : "\n");
  }
  Serial.print("Output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(i < output->dims->size - 1 ? " x " : "\n");
  }
  
  //Verifica que el modelo esté cuantizado en INT8
  if (input->type==kTfLiteInt8)
    Serial.println("Modelo está cuantizado en INT8");
  else {
    Serial.println("Modelo debe cuantizarse en INT8");
    while (true) {}
  }

  //Lleva la cuenta del número de inferencias calculadas
  inference_count = 0;
}


void loop() {
  delay(1000);

  // Captura un fotograma de 128x128 en formato RGB565, por lo tanto lo convierte a INT8
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Fallo en la captura de la cámara");
    return;
  }

  Serial.print("width:");
  Serial.println(fb->width);
  Serial.print("height:");
  Serial.println(fb->height);
  Serial.print("Format:");
  Serial.println(fb->format);
  Serial.print("Buflen:");
  Serial.println(fb->len);


  // Copia el buffer capturado por la cámara en el tensor de entrada. 
  // Realiza un proceso de normalización de la imagen.
  float scale = input->params.scale;
  int zero_point = input->params.zero_point;

  uint16_t *img = (uint16_t *)fb->buf;

  for (int i = 0; i < 128 * 128; i++) {
      uint16_t p = img[i];
      // Extrae RGB565
      uint8_t r = (p >> 11) & 0x1F;
      uint8_t g = (p >> 5)  & 0x3F;
      uint8_t b =  p        & 0x1F;

      // Expande a 8 bits
      r = (r << 3) | (r >> 2);
      g = (g << 2) | (g >> 4);
      b = (b << 3) | (b >> 2);

      // Normalización [0,1]
      float rf = r / 255.0f;
      float gf = g / 255.0f;
      float bf = b / 255.0f;

      // Cuantización
      input->data.int8[i * 3 + 0] =
          (int8_t)roundf(rf / scale) + zero_point;

      input->data.int8[i * 3 + 1] =
          (int8_t)roundf(gf / scale) + zero_point;

      input->data.int8[i * 3 + 2] =
          (int8_t)roundf(bf / scale) + zero_point;
  }

  esp_camera_fb_return(fb);

  // Inferencia
  int64_t t0 = esp_timer_get_time();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Falló la inferencia.");
    delay(1000);
    return;
  }
  int64_t t1 = esp_timer_get_time();
  size_t used_bytes = interpreter->arena_used_bytes();
  Serial.print("Arena usada (bytes):");
  Serial.println(used_bytes);
  Serial.print("Inferencia tomó (us)");
  Serial.println((long long)(t1 - t0));

  // Lee salida (2 clases: Healthy / Rejected)
  int best_idx = 0;
  float best_prob = 0;
  for (int i = 0; i < output->dims->data[1]; i++) {
      int8_t q = output->data.int8[i];
      float prob = (q - output->params.zero_point) * output->params.scale;
      if (prob > best_prob) {
        best_prob = prob;
        best_idx = i;
      }
      Serial.printf("Clase %d: %f\n", i, prob);
  }
  Serial.print("Clase predicha: ");
  Serial.println(best_idx == 0 ? "Healthy" : "Rejected");
}
