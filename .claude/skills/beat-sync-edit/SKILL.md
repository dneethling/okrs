---
name: beat-sync-edit
description: Analyse a music track (or a video's audio) for tempo, beat grid, bar downbeats and energy drops, then plan and verify video cuts and transitions that land on the beat. Use for showreels, sizzle reels, social edits, or any time cuts must sync to music. Needs only ffmpeg and numpy.
---

# Beat-synced editing

Never assume a track's tempo or cut "by feel". Measure the audio first, plan every cut on the measured grid, then verify the render.

## Workflow

1. **Analyse the track**
   `python3 -I scripts/beats.py analyze track.mp3 --out beats.json`
   Gives `bpm`, `beat_period`, `first_beat`, `beats[]` (seconds), `downbeats[]` (bar starts), `energy_drops[]` (bars that jump in energy: put hero shots and brand reveals here), `onsets[]` (strong hits, good for accents/flashes) and `bpm_candidates`.
   - If the BPM looks half or double what you expect, the truth is in `bpm_candidates`. Cut every 2 beats instead of every 1 (or the reverse).
   - Steady-tempo music only. Rubato, live drums or tempo changes will fit poorly; say so rather than guess.
2. **Plan shots in beats, not seconds**
   `python3 -I scripts/beats.py cuts beats.json 4,2,2,2,2,4,1,1,1,1` returns exact start/end times. Typical pacing: 4 beats (establish), 2 beats (rhythm), 1 beat (montage burst). Put brand/logo reveals on a downbeat or on an `energy_drops` bar. `snap` pulls arbitrary times onto the nearest beat.
3. **Build with ffmpeg.** Hard rules that make the cut feel tight:
   - Centre every transition on the beat. For an xfade of length T, start it at `beat - T/2`. Starting it on the beat makes the new shot land T/2 late, which reads as sloppy. Give each middle shot `D + T` of source, first and last `D + T/2`.
   - Keep transitions short (0.1 to 0.25 s) and vary them: slide, smooth wipe, circleopen, zoomin, fadewhite, hblur. Match the energy: fast montage gets fast transitions.
   - Add motion inside every shot (punch-in, pull-out, pan). Static shots die between cuts.
   - Mux the track last with `-shortest`; do not retime video to fit audio.
4. **Verify**
   `python3 -I scripts/beats.py verify out.mp4 --thresh 0.2`
   Detects the video's cuts and compares them with the beats of its own audio. Reports `on_beat_pct` and `median_offset_ms`.
   - Hard cuts should be within about 40 ms of a beat.
   - Soft transitions are detected near their midpoint, so a centred xfade should show a median near 0 and a transition that ends on the beat shows about -T/2. A consistent offset beyond that means shots are early or late: shift by `median_offset_ms`.
   - If it reports "no hard cuts", lower `--thresh`.

## How the analysis works (so you can trust or debug it)

Spectral-flux onset envelope with extra weight on the sub-200 Hz band, tempo from autocorrelation with a mild 120 BPM prior, then a comb-filter fit of tempo and phase over the whole track. The phase stage weights kick hits heavily, because off-beat hi-hats and bass can otherwise win and shift the whole grid by half a beat. A calibrated 26 ms lag corrects the STFT window stamp. Tested on a synthetic 128 BPM track: 128.00 BPM, first beat within 2 ms.

## Limits

- Detects tempo and downbeats, not melody, lyrics or mood. Pick section changes with `energy_drops` and your ears.
- Cannot hear the track for taste. Check the result is musically sensible, not just numerically on the beat.
- Never claim a cut is on the beat without running `verify` on the final render.
