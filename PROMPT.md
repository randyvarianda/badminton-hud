# Build prompt

The prompt that describes this tool as it stands now. Paste it into a new session to rebuild it or to start a variant. It is the original request, updated with every change made since.

---

Build me a tool that adds a game-style HUD to badminton clips (portrait, one rally). It should run with one command, `hud <clip>`, and put everything for that clip in a folder named after it, next to the clip: `<name>_HUD.mp4`, an error-check report, a config file and a cache of the slow steps. Test every step on the attached clip.

Test clip: [drag your clip here]
Game: doubles (4 players)
Players:
- Randy, blue shirt, camera's side
- Maria, magenta shirt, camera's side
- Alif, red shirt, far side
- Rizky, pink shirt, far side

Sides can swap between clips; the config file sets them.

Filming: best from behind the baseline, high up and on a tripod. Hand-held is fine for the automatic court finding. A fixed camera somewhere else (a corner or the side) is also supported, by giving court points by hand (step 4).

On screen:
- Foot rings that lie flat on the court.
- Name tags above heads, stacked so they don't overlap.
- A top bar with RALLY TIME / RALLY COUNT / SHOT SPEED. SHOT SPEED flashes on each new shot.
- Bottom-left: stamina and distance for all 4 players.
- Bottom-right: COURT MAP.
- Nothing drawn on the shuttle itself. No LOCK ON and no trail: both were tried and looked misplaced on real footage.
- Lato and the dark-mode colour tokens from my design system (`pd-design-artifacts`).

How:
1. **Prepare.** ffmpeg to 1080x1920 at the source frame rate. Extract the audio and find racket sounds (onsets in the 2–8 kHz band). Score each sound by its linear loudness, so quiet clicks from the next courts don't count.
2. **Camera motion.** Frame-to-frame homography from features on the ceiling and walls, with people masked out.
3. **People.** YOLO11m + ByteTrack on every frame, imgsz 1280 (CUDA/MPS if available). Cache the result.
4. **Court in the first frame.**
   - White floor lines: a top-hat on the darkest colour channel, inside the floor area.
   - Hough segments → vanishing point of the lines running away from the camera.
   - The two lowest long horizontal lines are the baseline (0 m) and the doubles long service line (0.76 m). If the baseline is cut off, try the other pairs (0 / 0.76 / 4.72 m).
   - Match the lines running away to x = 0, 0.46, 3.05, 5.64, 6.10 m with a cross-ratio fit → homography.
   - Keep the fit that covers the most white paint, and draw the court over the first frame so I can check it.
   - Fixed camera elsewhere: if the config has `[court] points` (4+ entries of `[x m, y m, pixel x, pixel y]` in the original clip's pixels), use those instead. Reuse them for every clip filmed from that spot.
5. **Court in every frame.** Predict from the camera motion, then snap onto the real white lines using the near half only. Re-read the lines from the picture every 2nd frame and keep whichever fit covers more white paint; chaining frame to frame alone makes the lines drift. With fixed court points, skip the re-read.
6. **Who is who.**
   - Torso colour by hue, using only the strongly coloured pixels, then re-centred on the clip's own lighting.
   - Side of the net (from the config).
   - Last position, only to break near-ties. Position alone swaps the names when partners switch sides.
   - Then one name per ByteTrack id.
   - Red and pink on the same side are close in colour: tell them apart by hue and brightness, and check those names carefully in the shot pictures.
7. **Shuttle.**
   - Small bright blobs that are brighter than the frames ±3 and ±10, outside the player boxes. Compensate the camera both by the background and by the floor plane.
   - Seed-and-grow tracklets, each scored by how bright it is against the clip's own clutter.
   - One path through the whole rally that may only change direction at the racket of one of our 4 players. This drops shuttles from the next courts.
8. **Shots.**
   - Candidates:
     - the shuttle reverses or changes speed at a racket
     - bright flights that start or end at one of our rackets (needed when the audio is silent)
     - racket sounds where the shuttle cannot be seen
   - A shot that is both seen and heard counts once, timed by the sound.
   - Shots always alternate sides; the hitter is the team member closer to the shuttle.
   - Launch speed = (e^(kD) − 1) / (kT), with k = 9.81 / 6.8², D = distance from the hitter to the next hitter (m), T = flight time.
   - The rally ends where the last shot's flight stops being seen.
9. **Draw the HUD** with Pillow and add the original audio back.
10. **Error report** (`index.html`):
    - court lines on 8 frames
    - a picture of every shot (at the hit and 0.2 s later)
    - pictures of long stretches without a shot
    - warnings: speeds above 250 or below 25 km/h, shots less than 0.33 s or more than 2.4 s apart, guessed hitters, weak court fits, unidentified players
    - how to fix each warning through the config file

    Config fixes:
    - `add_shots`: a time, or `{ time, hitter }`
    - `remove_shots`
    - `rally_end`

    Re-runs only repeat the quick steps.

Manual hit list: when the audio is silent, I'll send hits as QuickTime `seconds.frames` plus who hit, e.g. `5.07 Randy`. Convert them at the clip's frame rate into `add_shots` with hitters, remove the automatic shots that aren't on the list, and re-run.

Before you hand it over: look at the check pictures yourself every time, fix until they pass, then tell me what you are still unsure about.
