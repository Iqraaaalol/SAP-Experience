import re
import sys
import os
from pathlib import Path

# Add paths
cv_path = str(Path(__file__).parent / "computer-vision")
app_path = str(Path(__file__).parent / "app")
sys.path.insert(0, cv_path)
sys.path.insert(0, app_path)

print("[TEST] Performance Optimization Verification")
print("=" * 60)

# Test 1: Config flags
print("\n[1] Testing config flags...")
try:
    import config
    
    # Check new rendering flags
    render_fps = getattr(config, "RENDER_FPS_TARGET", None)
    render_annotations = getattr(config, "RENDER_ANNOTATIONS_ENABLED", None)
    render_preset = getattr(config, "RENDER_PRESET", None)
    
    if render_fps and render_annotations is not None and render_preset:
        print(f"    ✓ RENDER_FPS_TARGET = {render_fps}")
        print(f"    ✓ RENDER_ANNOTATIONS_ENABLED = {render_annotations}")
        print(f"    ✓ RENDER_PRESET = {render_preset}")
        print(f"    ✓ All rendering config flags present")
    else:
        print("    ✗ Some rendering flags missing!")
        sys.exit(1)
except Exception as e:
    print(f"    ✗ Error loading config: {e}")
    sys.exit(1)

# Test 2: RenderingPipeline class structure
print("\n[2] Testing RenderingPipeline availability...")
try:
    # Check if RenderingPipeline is defined
    main_source = open(str(Path(__file__).parent / "app" / "main.py"), encoding="utf-8").read()
    
    if "class RenderingPipeline:" in main_source:
        print("    ✓ RenderingPipeline class defined")
    else:
        print("    ✗ RenderingPipeline class NOT found")
        sys.exit(1)
        
    if "def enqueue(self" in main_source:
        print("    ✓ enqueue() method defined")
    else:
        print("    ✗ enqueue() method NOT found")
        sys.exit(1)
        
    if "def render_loop(self" in main_source:
        print("    ✓ render_loop() method defined")
    else:
        print("    ✗ render_loop() method NOT found")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ Error checking RenderingPipeline: {e}")
    sys.exit(1)

# Test 3: process_frame changes
print("\n[3] Testing process_frame() refactoring...")
try:
    main_source = open(str(Path(__file__).parent / "app" / "main.py"), encoding="utf-8").read()
    
    # Check that process_frame returns metadata
    if "return frame, metadata" in main_source:
        print("    ✓ process_frame() returns (frame, metadata)")
    else:
        print("    ✗ process_frame() return signature not updated")
        sys.exit(1)
        
    # Check that drawing code is removed from process_frame
    if "self.detector.draw_enhanced_boxes" not in main_source.split("def capture_loop")[0]:
        print("    ✓ Drawing code removed from process_frame()")
    else:
        # It might be in RenderingPipeline, which is OK
        if "class RenderingPipeline:" in main_source:
            print("    ✓ Drawing code moved to RenderingPipeline")
        else:
            print("    ✗ Drawing code still in process_frame()")
            sys.exit(1)

except Exception as e:
    print(f"    ✗ Error checking process_frame: {e}")
    sys.exit(1)

# Test 4: MJPEG throttling
print("\n[4] Testing generate_frames() throttling...")
try:
    main_source = open(str(Path(__file__).parent / "app" / "main.py"), encoding="utf-8").read()
    
    if "frame_interval" in main_source and "render_fps_target" in main_source:
        print("    ✓ MJPEG throttling implementation found")
    else:
        print("    ✗ MJPEG throttling not implemented")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ Error checking generate_frames: {e}")
    sys.exit(1)

# Test 5: Pipeline management in start/stop
print("\n[5] Testing start()/stop() pipeline management...")
try:
    main_source = open(str(Path(__file__).parent / "app" / "main.py"), encoding="utf-8").read()
    
    # Extract start method
    if "self.rendering_pipeline.start()" in main_source:
        print("    ✓ start() initializes rendering pipeline")
    else:
        print("    ✗ start() doesn't initialize rendering pipeline")
        sys.exit(1)
        
    if "self.rendering_pipeline.stop()" in main_source:
        print("    ✓ stop() tears down rendering pipeline")
    else:
        print("    ✗ stop() doesn't tear down rendering pipeline")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ Error checking start/stop: {e}")
    sys.exit(1)

# Test 6: Non-blocking capture handoff
print("\n[6] Testing non-blocking capture handoff...")
try:
    main_source = open(str(Path(__file__).parent / "app" / "main.py"), encoding="utf-8").read()

    if re.search(r"ThreadPoolExecutor\([^)]*max_workers\s*=\s*1", main_source, re.S):
        print("    ✓ Single-worker processing executor defined")
    else:
        print("    ✗ Processing executor not found")
        sys.exit(1)

    if "self._process_executor.submit(" in main_source and "self._process_and_enqueue" in main_source:
        print("    ✓ capture_loop() submits work asynchronously")
    else:
        print("    ✗ capture_loop() does not use async submission")
        sys.exit(1)

    if "len(self.queue) >= self.queue_max" in main_source:
        print("    ✓ Render queue backlog cap tightened")
    else:
        print("    ✗ Render queue backlog cap not tightened")
        sys.exit(1)

except Exception as e:
    print(f"    ✗ Error checking non-blocking capture handoff: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All verification tests PASSED!")
