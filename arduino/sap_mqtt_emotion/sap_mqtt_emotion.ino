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
const char MQTT_TOPIC_ALL[]      = "sap/seats/emotion";        // combined summary
const char MQTT_TOPIC_SEAT[]     = "sap/seats/emotion/+";      // per-seat wildcard
const char MQTT_TOPIC_OVERRIDE[] = "sap/lighting/override";    // crew manual override

// Seat IDs we care about
const int  NUM_SEATS = 4;
const char* SEAT_IDS[NUM_SEATS] = {"1A", "1B", "2A", "2B"};

// WS2812B LED strip
#define LED_PIN    6      // DIN data pin
#define LED_COUNT  30     // number of LEDs in the strip
// effect IDs: 0=solid  1=slow pulse(~4s)  2=fade-in(~3s)  3=ultra-slow pulse(~8s)

// How often to print status (ms)
const unsigned long STATUS_INTERVAL = 5000;

// Transition / guard timings
const unsigned long TRANSITION_MS = 5000;   // crossfade duration between scenes
const unsigned long SUSTAIN_MS    = 15000;  // new emotion must hold this long before switching
const unsigned long COOLDOWN_MS   = 15000;  // lock-out period after a switch completes
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

MoodScene MOOD_SAD      = { 40, 50, 150, 0,   // soft blue with tiny red warmth
  "SAD",      "Soft blue. Emotional safety. Warm enough to not feel clinical." };

MoodScene MOOD_ANXIOUS  = { 60,  10,  180, 3,   // lavender + ultra-slow pulse
  "ANXIOUS",  "Lavender. Slows breathing subconsciously. Reduces perceived threat." };

// ── Aggregation tables (order must stay consistent) ────────────
// All addressable mood buckets
const int  NUM_MOODS = 6;
MoodScene* ALL_MOODS[NUM_MOODS] = {
  &MOOD_HAPPY, &MOOD_NEUTRAL, &MOOD_STRESSED,
  &MOOD_ANGRY, &MOOD_SAD,     &MOOD_ANXIOUS
};
// Priority multipliers — urgent / negative emotions outweigh mild
// ones when raw confidence scores are close across seats.
const float MOOD_PRIORITY[NUM_MOODS] = {
  1.0f,  // HAPPY    — reinforce but don't over-index
  1.0f,  // NEUTRAL  — baseline weight
  1.5f,  // STRESSED — needs intervention
  1.5f,  // ANGRY    — needs intervention
  1.3f,  // SAD      — moderate priority
  1.5f,  // ANXIOUS  — needs intervention
};
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

// Crossfade state
uint8_t       fromR = 0, fromG = 0, fromB = 0;
bool          isTransitioning = false;
unsigned long transitionStart = 0;

// Change-rate guard
MoodScene*    pendingMood    = nullptr;
unsigned long pendingStart   = 0;
unsigned long lastSwitchTime = 0;

// Manual override (crew dashboard)
bool       overrideActive = false;
MoodScene* overrideScene  = &MOOD_NEUTRAL;

// ---------------------------------------------------------------
// Map a scene name string (e.g. "HAPPY") → MoodScene*
// Used by the crew dashboard manual override.
// ---------------------------------------------------------------
MoodScene* sceneNameToMood(const char* name) {
  if      (strcmp(name, "HAPPY")    == 0) return &MOOD_HAPPY;
  else if (strcmp(name, "NEUTRAL")  == 0) return &MOOD_NEUTRAL;
  else if (strcmp(name, "STRESSED") == 0) return &MOOD_STRESSED;
  else if (strcmp(name, "ANGRY")    == 0) return &MOOD_ANGRY;
  else if (strcmp(name, "SAD")      == 0) return &MOOD_SAD;
  else if (strcmp(name, "ANXIOUS")  == 0) return &MOOD_ANXIOUS;
  else                                    return &MOOD_NEUTRAL;
}

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

  // Subscribe to crew dashboard lighting override
  mqttClient.subscribe(MQTT_TOPIC_OVERRIDE);
  Serial.print("Subscribed to: ");
  Serial.println(MQTT_TOPIC_OVERRIDE);
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

  // ── Crew dashboard lighting override ────────────────────────
  if (topic == String(MQTT_TOPIC_OVERRIDE)) {
    JsonDocument ov;
    if (deserializeJson(ov, payload) == DeserializationError::Ok) {
      bool enable = ov["enabled"] | false;
      const char* sceneName = ov["scene"] | "NEUTRAL";
      overrideActive = enable;
      if (enable) {
        overrideScene = sceneNameToMood(sceneName);
        pendingMood   = nullptr;          // discard any pending auto transition
        applyMood(overrideScene);         // immediate crossfade to chosen scene
        Serial.print("[OVERRIDE] ON → ");
        Serial.println(overrideScene->name);
      } else {
        overrideActive = false;
        // Snap back to whatever the cabin aggregation currently says
        applyMood(aggregateCabinMood());
        Serial.println("[OVERRIDE] OFF → returning to auto mode");
      }
    }
    return;
  }
  // ────────────────────────────────────────────────────────────

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
    // Aggregate all updated seats into a cabin-wide decision
    evaluateMoodCandidate(aggregateCabinMood());
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
  else if (strcmp(emotion, "sad")      == 0)                          return &MOOD_SAD;
  else if (strcmp(emotion, "anxious")  == 0 ||
           strcmp(emotion, "fear")     == 0)                          return &MOOD_ANXIOUS;
  else if (strcmp(emotion, "disgust")  == 0)                          return &MOOD_ANGRY;
  else                                                                return &MOOD_NEUTRAL;
}

// ---------------------------------------------------------------
// Weighted aggregation across all seats that have data.
// Each seat contributes: confidence × MOOD_PRIORITY to its bucket.
// Returns the MoodScene* with the highest total weighted score.
// ---------------------------------------------------------------
MoodScene* aggregateCabinMood() {
  float scores[NUM_MOODS] = {};
  int   activeSeats = 0;

  for (int i = 0; i < NUM_SEATS; i++) {
    if (seatEmotions[i].length() == 0) continue;
    activeSeats++;
    MoodScene* mood = emotionToMood(seatEmotions[i].c_str());
    for (int m = 0; m < NUM_MOODS; m++) {
      if (ALL_MOODS[m] == mood) {
        scores[m] += seatConfidences[i] * MOOD_PRIORITY[m];
        break;
      }
    }
  }

  if (activeSeats == 0) return &MOOD_NEUTRAL;

  // Log weighted scores for monitoring
  Serial.println("[AGG] Cabin weighted scores:");
  int best = 0;
  for (int m = 0; m < NUM_MOODS; m++) {
    if (scores[m] > 0.0f) {
      Serial.print("      ");
      Serial.print(ALL_MOODS[m]->name);
      Serial.print(": ");
      Serial.println(scores[m], 3);
    }
    if (scores[m] > scores[best]) best = m;
  }
  Serial.print("[AGG] Winner → ");
  Serial.println(ALL_MOODS[best]->name);
  return ALL_MOODS[best];
}

// ---------------------------------------------------------------
// Sustain + cooldown guard — accepts the aggregated cabin winner.
// Called from handleEmotion() (per-seat) and the combined summary
// branch so both code paths share identical guard logic.
// ---------------------------------------------------------------
void evaluateMoodCandidate(MoodScene* candidate) {
  // Manual override is active — crew is in control, ignore auto-detection
  if (overrideActive) return;

  unsigned long now = millis();

  // Already in this scene — nothing to do
  if (candidate == currentMood) {
    pendingMood = nullptr;
    return;
  }

  // New candidate — (re)start the sustain window
  if (candidate != pendingMood) {
    pendingMood  = candidate;
    pendingStart = now;
    Serial.print("[GUARD] Sustain started for ");
    Serial.println(candidate->name);
    return;
  }

  // Same candidate still pending — check sustain window
  if (now - pendingStart < SUSTAIN_MS) {
    Serial.print("[GUARD] Sustain remaining: ");
    Serial.print((SUSTAIN_MS - (now - pendingStart)) / 1000);
    Serial.println(" s");
    return;
  }

  // Sustain passed — check cooldown since last switch
  if (now - lastSwitchTime < COOLDOWN_MS) {
    Serial.print("[GUARD] Cooldown remaining: ");
    Serial.print((COOLDOWN_MS - (now - lastSwitchTime)) / 1000);
    Serial.println(" s");
    return;
  }

  // All guards passed — commit the new scene
  applyMood(candidate);
}

// ---------------------------------------------------------------
// Set a new active mood scene
// ---------------------------------------------------------------
void applyMood(MoodScene* mood) {
  if (mood == currentMood) return;

  // Snapshot the colour currently rendered on pixel 0 as the crossfade start
  uint32_t cur = strip.getPixelColor(0);
  fromR = (uint8_t)((cur >> 16) & 0xFF);
  fromG = (uint8_t)((cur >>  8) & 0xFF);
  fromB = (uint8_t)((cur      ) & 0xFF);

  currentMood     = mood;
  isTransitioning = true;
  transitionStart = millis();
  fadeInDone      = false;
  pendingMood     = nullptr;
  lastSwitchTime  = millis();

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
  // ── Crossfade to new scene ─────────────────────────────────
  if (isTransitioning) {
    float t = (float)(millis() - transitionStart) / (float)TRANSITION_MS;
    if (t >= 1.0f) {
      // Transition complete — hand off to normal effect engine
      isTransitioning = false;
      effectStart     = millis();
      t = 1.0f;
    }
    uint8_t r = (uint8_t)(fromR + t * ((int)currentMood->r - (int)fromR));
    uint8_t g = (uint8_t)(fromG + t * ((int)currentMood->g - (int)fromG));
    uint8_t b = (uint8_t)(fromB + t * ((int)currentMood->b - (int)fromB));
    strip.fill(strip.Color(r, g, b));
    strip.show();
    return;
  }
  // ────────────────────────────────────────────────────────────

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
// React to a per-seat emotion update.
// Seat data is already stored in seatEmotions[] before this call.
// Aggregates all seats into a single cabin-wide candidate and
// passes it through the sustain + cooldown guard.
// ---------------------------------------------------------------
void handleEmotion(const char* seatId, const char* emotion, float confidence) {
  evaluateMoodCandidate(aggregateCabinMood());
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
