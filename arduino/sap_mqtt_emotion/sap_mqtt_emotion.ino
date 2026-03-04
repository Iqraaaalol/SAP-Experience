#include <Adafruit_NeoPixel.h>

/*
 * SAP Seat Emotion MQTT Receiver — Arduinoq UNO R4 WiFi
 *
 * Subscribes to MQTT topics published by the Python emotion detector:
 *   - sap/seats/emotion/{seatId}   (per-seat JSON messages)
 *   - sap/seats/emotion            (combined summary)
 *
 * Each per-seat message is JSON:
 *   {"seat": "1A", "emotion": "happy", "confidence": 0.92}
 *
 * Hardware:
 *   - Arduino UNO R4 WiFi (built-in ESP32-S3 based WiFi)
 *
 * Libraries required (install via Library Manager):
 *   - WiFiS3            (built-in with UNO R4 WiFi board package)
 *   - ArduinoMqttClient
 *   - ArduinoJson       (v7+)
 *   - Adafruit NeoPixel (for WS2812B strip)
 *
 * Board package:
 *   - Arduino UNO R4 Boards  (via Boards Manager)
 */

#include <WiFiS3.h>
#include <ArduinoMqttClient.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>

// ======================== CONFIGURATION ========================
// WiFi credentials — update these for your network
const char WIFI_SSID[]     = "where_is_sushi?";
const char WIFI_PASSWORD[] = "Sushi@2005";

// MQTT broker — use the IP of the machine running Mosquitto
const char MQTT_BROKER[]   = "10.110.0.13";  // <-- Change to your broker IP
const int  MQTT_PORT       = 1883;

// MQTT topic to subscribe to (matches Python publisher)
const char MQTT_TOPIC_ALL[]  = "sap/seats/emotion";    // combined summary
const char MQTT_TOPIC_SEAT[] = "sap/seats/emotion/+";  // per-seat wildcard

// Seat IDs we care about
const int  NUM_SEATS = 4;
const char* SEAT_IDS[NUM_SEATS] = {"1A", "1B", "2A", "2B"};

// WS2812B LED strip
#define LED_PIN    6      // DIN data pin
#define LED_COUNT  30     // number of LEDs in the strip
// effect IDs: 0=solid  1=slow pulse(~4s)  2=fade-in(~3s)  3=ultra-slow pulse(~8s)

// How often to print status (ms)
const unsigned long STATUS_INTERVAL = 5000;
// ===============================================================

// ======================== MOOD SCENES =========================
struct MoodScene {
  uint8_t     r, g, b;
  uint8_t     effect;        // 0=solid | 1=slow pulse | 2=fade-in | 3=ultra-slow pulse
  const char* name;
  const char* description;
};

MoodScene MOOD_HAPPY    = { 30,  200, 255, 0,
  "HAPPY",    "Bright cyan-white. Daylight energy. Reinforce positive state." };

MoodScene MOOD_NEUTRAL  = { 80,  80,  100, 0,
  "NEUTRAL",  "Cool white. Non-interventional baseline. Standard cruise." };

MoodScene MOOD_STRESSED = { 30,  20,  200, 1,   // blue-violet + slow pulse
  "STRESSED", "Blue-violet. Activates parasympathetic. Lowers cortisol. Slow pulse." };

MoodScene MOOD_ANGRY    = { 255, 30,  85,  0,   // warm pink — NOT red
  "ANGRY",    "Warm pink. Baker-Miller effect. Involuntary aggression reduction." };

MoodScene MOOD_FATIGUE  = { 0,   180, 140, 2,   // teal-green + fade-in
  "FATIGUE",  "Teal-green. Promotes melatonin. Evening sky signal. Helps sleep." };

MoodScene MOOD_SAD      = { 20,  40,  200, 0,   // soft blue with tiny red warmth
  "SAD",      "Soft blue. Emotional safety. Warm enough to not feel clinical." };

MoodScene MOOD_ANXIOUS  = { 60,  10,  180, 3,   // lavender + ultra-slow pulse
  "ANXIOUS",  "Lavender. Slows breathing subconsciously. Reduces perceived threat." };

// ===============================================================

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

WiFiClient     wifiClient;
MqttClient     mqttClient(wifiClient);

// Store latest emotion per seat
String seatEmotions[NUM_SEATS];
float  seatConfidences[NUM_SEATS];

unsigned long lastStatusPrint = 0;

// Effect state
MoodScene*    currentMood  = &MOOD_NEUTRAL;
unsigned long effectStart  = 0;
bool          fadeInDone   = false;

// ---------------------------------------------------------------
// Find the index of a seat ID in our array (-1 if not found)
// ---------------------------------------------------------------
int seatIndex(const char* id) {
  for (int i = 0; i < NUM_SEATS; i++) {
    if (strcmp(SEAT_IDS[i], id) == 0) return i;
  }
  return -1;
}

// ---------------------------------------------------------------
// Connect (or reconnect) to WiFi
// ---------------------------------------------------------------
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connection FAILED — will retry...");
  }
}

// ---------------------------------------------------------------
// Connect (or reconnect) to MQTT broker
// ---------------------------------------------------------------
void connectMQTT() {
  if (mqttClient.connected()) return;

  Serial.print("Connecting to MQTT broker: ");
  Serial.print(MQTT_BROKER);
  Serial.print(":");
  Serial.println(MQTT_PORT);

  if (!mqttClient.connect(MQTT_BROKER, MQTT_PORT)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());
    return;
  }

  Serial.println("MQTT connected!");

  // Subscribe to per-seat wildcard topic
  mqttClient.subscribe(MQTT_TOPIC_SEAT);
  Serial.print("Subscribed to: ");
  Serial.println(MQTT_TOPIC_SEAT);

  // Also subscribe to the combined summary topic
  mqttClient.subscribe(MQTT_TOPIC_ALL);
  Serial.print("Subscribed to: ");
  Serial.println(MQTT_TOPIC_ALL);
}

// ---------------------------------------------------------------
// Called when an MQTT message arrives
// ---------------------------------------------------------------
void onMqttMessage(int messageSize) {
  String topic = mqttClient.messageTopic();

  // Read the payload
  String payload = "";
  while (mqttClient.available()) {
    payload += (char)mqttClient.read();
  }

  // Check if this is a per-seat message (topic ends with /1A, /1B, etc.)
  // or the combined summary
  int lastSlash = topic.lastIndexOf('/');
  String suffix = topic.substring(lastSlash + 1);

  // Try to parse as per-seat message
  JsonDocument doc;
  DeserializationError error = deserializeJson(doc, payload);

  if (error) {
    Serial.print("JSON parse error: ");
    Serial.println(error.c_str());
    return;
  }

  // Per-seat message: {"seat": "1A", "emotion": "happy", "confidence": 0.92}
  if (doc.containsKey("seat")) {
    const char* seat    = doc["seat"];
    const char* emotion = doc["emotion"];
    float confidence    = doc["confidence"];

    int idx = seatIndex(seat);
    if (idx >= 0) {
      seatEmotions[idx]    = String(emotion);
      seatConfidences[idx] = confidence;
    }

    Serial.print("[SEAT ");
    Serial.print(seat);
    Serial.print("] ");
    Serial.print(emotion);
    Serial.print(" (");
    Serial.print(confidence, 2);
    Serial.println(")");

    // ====== ACT ON EMOTION HERE ======
    // Example: drive an LED, servo, buzzer, or LED matrix based on emotion
    handleEmotion(seat, emotion, confidence);
  }
  // Combined summary: {"1A": {"emotion": "happy", ...}, "1B": {...}}
  else {
    for (int i = 0; i < NUM_SEATS; i++) {
      if (doc.containsKey(SEAT_IDS[i])) {
        const char* emotion = doc[SEAT_IDS[i]]["emotion"];
        float confidence    = doc[SEAT_IDS[i]]["confidence"];
        seatEmotions[i]    = String(emotion);
        seatConfidences[i] = confidence;
      }
    }
  }
}

// ---------------------------------------------------------------
// Map emotion string → MoodScene
// ---------------------------------------------------------------
MoodScene* emotionToMood(const char* emotion) {
  if      (strcmp(emotion, "happy")    == 0)                          return &MOOD_HAPPY;
  else if (strcmp(emotion, "neutral")  == 0)                          return &MOOD_NEUTRAL;
  else if (strcmp(emotion, "stress")   == 0 ||
           strcmp(emotion, "stressed") == 0)                          return &MOOD_STRESSED;
  else if (strcmp(emotion, "angry")    == 0)                          return &MOOD_ANGRY;
  else if (strcmp(emotion, "fatigue")  == 0 ||
           strcmp(emotion, "sleeping") == 0)                          return &MOOD_FATIGUE;
  else if (strcmp(emotion, "sad")      == 0)                          return &MOOD_SAD;
  else if (strcmp(emotion, "anxious")  == 0 ||
           strcmp(emotion, "fear")     == 0)                          return &MOOD_ANXIOUS;
  else if (strcmp(emotion, "disgust")  == 0)                          return &MOOD_ANGRY;
  else                                                                return &MOOD_NEUTRAL;
}

// ---------------------------------------------------------------
// Set a new active mood scene
// ---------------------------------------------------------------
void applyMood(MoodScene* mood) {
  if (mood == currentMood) return;
  currentMood = mood;
  effectStart = millis();
  fadeInDone  = false;
  Serial.print("[LIGHT] → ");
  Serial.print(mood->name);
  Serial.print(" | ");
  Serial.println(mood->description);
}

// ---------------------------------------------------------------
// Drive the WS2812B strip according to the active mood effect
// Called every loop() iteration
// ---------------------------------------------------------------
void updateEffect() {
  unsigned long elapsed = millis() - effectStart;
  float         bright;

  switch (currentMood->effect) {

    case 0:  // Solid colour
      strip.fill(strip.Color(currentMood->r, currentMood->g, currentMood->b));
      strip.show();
      break;

    case 1:  // Slow pulse — ~4 s period, brightness 15 %–100 %
      bright = (sinf(TWO_PI * elapsed / 4000.0f) + 1.0f) * 0.5f;
      bright = 0.15f + bright * 0.85f;
      strip.fill(strip.Color(
        (uint8_t)(currentMood->r * bright),
        (uint8_t)(currentMood->g * bright),
        (uint8_t)(currentMood->b * bright)));
      strip.show();
      break;

    case 2:  // Fade-in over ~3 s, then hold solid
      if (!fadeInDone) {
        bright = min(1.0f, (float)elapsed / 3000.0f);
        if (bright >= 1.0f) fadeInDone = true;
        strip.fill(strip.Color(
          (uint8_t)(currentMood->r * bright),
          (uint8_t)(currentMood->g * bright),
          (uint8_t)(currentMood->b * bright)));
        strip.show();
      }
      // Once faded in, colour is held — no further update needed
      break;

    case 3:  // Ultra-slow pulse — ~8 s period, brightness 10 %–100 %
      bright = (sinf(TWO_PI * elapsed / 8000.0f) + 1.0f) * 0.5f;
      bright = 0.10f + bright * 0.90f;
      strip.fill(strip.Color(
        (uint8_t)(currentMood->r * bright),
        (uint8_t)(currentMood->g * bright),
        (uint8_t)(currentMood->b * bright)));
      strip.show();
      break;
  }
}

// ---------------------------------------------------------------
// React to an emotion — maps to a WS2812B MoodScene
// ---------------------------------------------------------------
void handleEmotion(const char* seatId, const char* emotion, float confidence) {
  applyMood(emotionToMood(emotion));
}

// ---------------------------------------------------------------
// Print a status summary of all seats
// ---------------------------------------------------------------
void printStatus() {
  Serial.println("──────────── Seat Status ────────────");
  for (int i = 0; i < NUM_SEATS; i++) {
    Serial.print("  Seat ");
    Serial.print(SEAT_IDS[i]);
    Serial.print(": ");
    if (seatEmotions[i].length() > 0) {
      Serial.print(seatEmotions[i]);
      Serial.print(" (");
      Serial.print(seatConfidences[i], 2);
      Serial.println(")");
    } else {
      Serial.println("— no data —");
    }
  }
  Serial.println("─────────────────────────────────────");
}

// ===============================================================
void setup() {
  Serial.begin(9600);
  // Do NOT block on Serial — without this fix the strip never lights up
  // if no Serial Monitor is open.
  delay(1000);  // short settle delay for power-on stability

  // WS2812B strip initialisation
  strip.begin();
  strip.setBrightness(200);  // 0-255; explicit so behaviour is always predictable
  strip.clear();
  strip.show();

  // --- Startup flash: white for 500 ms so you can confirm wiring is good ---
  strip.fill(strip.Color(200, 200, 200));
  strip.show();
  delay(1000);
  strip.clear();
  strip.show();
  delay(200);
  // -----------------------------------------------------------------

  // Start in NEUTRAL scene
  currentMood = &MOOD_NEUTRAL;
  effectStart = millis();

  Serial.println();
  Serial.println("================================");
  Serial.println(" SAP Seat Emotion MQTT Receiver");
  Serial.println("================================");

  // Initialize stored emotions
  for (int i = 0; i < NUM_SEATS; i++) {
    seatEmotions[i]    = "";
    seatConfidences[i] = 0.0;
  }

  connectWiFi();

  // Set MQTT message callback
  mqttClient.onMessage(onMqttMessage);

  connectMQTT();
}

// ===============================================================
void loop() {
  // Maintain connections
  connectWiFi();
  connectMQTT();

  // Process incoming MQTT messages
  mqttClient.poll();

  // Drive WS2812B lighting effect for the current mood
  updateEffect();

  // Periodic status print
  unsigned long now = millis();
  if (now - lastStatusPrint >= STATUS_INTERVAL) {
    lastStatusPrint = now;
    printStatus();
  }
}
