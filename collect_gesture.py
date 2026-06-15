"""
collect_gesture.py
==================
Simple webcam recorder to collect new gesture training videos.

Usage:
    python collect_gesture.py --gesture PLEASE --clips 10

This will create:
    ASL_dynamic/PLEASE/clip_001.avi
    ASL_dynamic/PLEASE/clip_002.avi
    ...

After collecting, run:
    python data_preprocessing.py
    python train_model.py
"""

import cv2
import os
import argparse
import time

# ── Config ────────────────────────────────────────────────────────────────────
CLIP_SECONDS   = 2      # How long (seconds) each clip is recorded
FPS            = 15     # Frame rate for saved clips
FRAME_W, FRAME_H = 640, 480
COUNTDOWN_SECS = 3      # Countdown before each clip starts

def record_gesture(gesture_name: str, num_clips: int):
    out_dir = os.path.join("ASL_dynamic", gesture_name.upper())
    os.makedirs(out_dir, exist_ok=True)

    # Find the next clip number to avoid overwriting existing clips
    existing = [f for f in os.listdir(out_dir) if f.endswith(".avi")]
    start_idx = len(existing) + 1

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    print(f"\n🎬 Recording gesture: '{gesture_name.upper()}'")
    print(f"   Clips to record : {num_clips}")
    print(f"   Duration each   : {CLIP_SECONDS}s")
    print(f"   Output folder   : {out_dir}")
    print(f"\n   Press SPACE to start each clip, 'q' to quit early.\n")

    clip_idx = start_idx
    clips_done = 0

    while clips_done < num_clips:
        # ── Wait for SPACE to start next clip ─────────────────────────────────
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                break
            display = cv2.flip(frame, 1)
            cv2.putText(display,
                        f"Clip {clips_done + 1}/{num_clips}: '{gesture_name.upper()}'",
                        (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 200), 2)
            cv2.putText(display, "Press SPACE when ready to record",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            cv2.putText(display, f"Clips recorded: {clips_done}",
                        (20, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            cv2.imshow("Gesture Collector", display)

            key = cv2.waitKey(10) & 0xFF
            if key == 32:   # SPACE
                waiting = False
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print(f"\n✅ Done. Recorded {clips_done} clips.")
                return

        # ── Countdown ─────────────────────────────────────────────────────────
        countdown_start = time.time()
        while True:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECS - int(elapsed)
            if remaining <= 0:
                break
            ret, frame = cap.read()
            if not ret:
                break
            display = cv2.flip(frame, 1)
            cv2.putText(display, str(remaining),
                        (FRAME_W // 2 - 30, FRAME_H // 2 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 80, 255), 5)
            cv2.putText(display, "GET READY!", (FRAME_W // 2 - 80, FRAME_H // 2 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.imshow("Gesture Collector", display)
            cv2.waitKey(10)

        # ── Record clip ───────────────────────────────────────────────────────
        clip_name = f"clip_{clip_idx:03d}.avi"
        clip_path = os.path.join(out_dir, clip_name)
        fourcc    = cv2.VideoWriter_fourcc(*"XVID")
        writer    = cv2.VideoWriter(clip_path, fourcc, FPS, (FRAME_W, FRAME_H))

        record_start = time.time()
        frames_written = 0

        print(f"  🔴 Recording clip {clips_done + 1}/{num_clips} → {clip_name}")

        while (time.time() - record_start) < CLIP_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1

            display = cv2.flip(frame, 1)
            time_left = CLIP_SECONDS - (time.time() - record_start)
            cv2.putText(display, f"🔴 RECORDING  {time_left:.1f}s",
                        (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(display, gesture_name.upper(),
                        (20, FRAME_H - 20), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 100), 3)
            cv2.imshow("Gesture Collector", display)
            cv2.waitKey(1)

        writer.release()
        clips_done += 1
        clip_idx   += 1
        print(f"     ✔ Saved ({frames_written} frames) → {clip_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! Recorded {clips_done} clips for '{gesture_name.upper()}'.")
    print(f"   Now run:  python data_preprocessing.py  →  python train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gesture training clips.")
    parser.add_argument("--gesture", required=True,
                        help="Name of the gesture (e.g. PLEASE, HELP, MORE, STOP)")
    parser.add_argument("--clips", type=int, default=10,
                        help="Number of clips to record (default: 10)")
    args = parser.parse_args()
    record_gesture(args.gesture, args.clips)
