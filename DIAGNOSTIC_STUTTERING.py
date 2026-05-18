#!/usr/bin/env python3
"""
Diagnostic script to identify webpage video stuttering bottleneck.
Enables all debug logging and shows queue depth, FPS, and timing.
"""

import sys
from pathlib import Path

# Add paths
cv_path = str(Path(__file__).parent / "computer-vision")
sys.path.insert(0, cv_path)

import config

print("=" * 70)
print("CV PERFORMANCE DIAGNOSTIC")
print("=" * 70)

print("\n[CURRENT SETTINGS]")
print(f"  RENDER_FPS_TARGET           = {config.RENDER_FPS_TARGET}")
print(f"  RENDER_ANNOTATIONS_ENABLED  = {config.RENDER_ANNOTATIONS_ENABLED}")
print(f"  RENDER_PRESET               = {config.RENDER_PRESET}")
print(f"  RENDER_LANDMARKS            = {config.RENDER_LANDMARKS}")
print(f"  RENDER_DASHBOARD            = {config.RENDER_DASHBOARD}")
print(f"  RENDER_SEAT_ZONES           = {config.RENDER_SEAT_ZONES}")
print(f"  JPEG_QUALITY                = {config.JPEG_QUALITY}")
print(f"  DEBUG_RENDERING_FPS         = {config.DEBUG_RENDERING_FPS}")
print(f"  SKIP_FRAMES_IF_BACKLOG      = {config.SKIP_FRAMES_IF_BACKLOG}")
print(f"  RENDER_QUEUE_MAX_BACKLOG    = {config.RENDER_QUEUE_MAX_BACKLOG}")

print("\n" + "=" * 70)
print("DIAGNOSTICS: Enable debug logging to find bottleneck")
print("=" * 70)

print("""
To diagnose webpage stuttering, follow these steps:

STEP 1: Enable Debug Logging
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Edit computer-vision/config.py and set:
  DEBUG_RENDERING_FPS = True  ← This will log rendering queue and FPS

STEP 2: Start CV from Crew Dashboard  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In Crew Dashboard, click "Enable CV" and watch the console output.

STEP 3: Interpret the Logs (in console where app is running)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You'll see output like:
  [RENDER] FPS: 15.2, Queue backlog: 0
  [RENDER] FPS: 14.8, Queue backlog: 1
  [RENDER] FPS: 12.1, Queue backlog: 3   ← PROBLEM: Queue at max!

ANALYSIS:

A) If Queue Backlog Stays at 0-1:
   ✅ Rendering pipeline is fine
   ❌ Problem is elsewhere (MJPEG encoding, network, browser)
   
   → Try: Lower JPEG_QUALITY from 85 to 75 (reduces encode time)

B) If Queue Backlog Grows to 2-3:
   ❌ Rendering is slower than capture
   
   Try (in order):
   1. Set RENDER_PRESET = 'low'      (minimal drawing)
   2. Set RENDER_LANDMARKS = False   (already default)
   3. Set RENDER_DASHBOARD = False   (skip dashboard)
   4. Set RENDER_SEAT_ZONES = False  (skip zone drawing)

C) If Queue Backlog = 0 but FPS drops from 15 to 8-10:
   ❌ MJPEG encoding is slow
   
   Try:
   1. Lower JPEG_QUALITY to 70
   2. Lower RENDER_FPS_TARGET from 15 to 12
   3. Check if browser tab is in focus (some browsers throttle hidden tabs)

D) If CPU usage spikes when stutter happens:
   ❌ Drawing operations are expensive
   
   Try: Profile which drawing operation is slow:
   - Set RENDER_DASHBOARD = False, test
   - Set RENDER_SEAT_ZONES = False, test  
   - Set RENDER_ANNOTATIONS_ENABLED = False (test raw frames)

STEP 4: Quick Tests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test A - Minimal Rendering (eliminate drawing as bottleneck):
  RENDER_ANNOTATIONS_ENABLED = False
  → If smooth: problem is drawing
  → If still stutters: problem is encoding/network

Test B - Lower Quality (reduce encoding time):
  JPEG_QUALITY = 60
  RENDER_FPS_TARGET = 12
  → If smooth: encoding was bottleneck
  → If still stutters: network/browser issue

Test C - Debug Queue Depth (shows backlog):
  DEBUG_RENDERING_FPS = True
  → Check console every 2 seconds
  → If backlog = 0: rendering is OK
  → If backlog = 3: rendering is too slow

STEP 5: Apply Best Settings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Once you identify the issue, apply permanent fix:

If rendering too slow:
  RENDER_PRESET = 'low'              # Most aggressive
  RENDER_LANDMARKS = False           # Already recommended
  
If MJPEG encoding too slow:
  JPEG_QUALITY = 75                  # Balance quality/speed
  
If network bottleneck:
  RENDER_FPS_TARGET = 12             # Lower FPS = less bandwidth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INSIGHT:
The fact that mood_detection.py runs smoothly means:
✅ Emotion detection is fast
✅ Face detection is fast  
✅ Processing logic is fine

But MJPEG rendering is slower because:
❌ Network latency to browser
❌ Frame encoding takes time
❌ Drawing operations aren't fast enough

Start with DEBUG_RENDERING_FPS = True and let's identify the culprit!
""")

print("\n" + "=" * 70)
print("NEXT: Set DEBUG_RENDERING_FPS = True in config.py, start CV, and")
print("check the console output to identify the exact bottleneck.")
print("=" * 70)
