# /brag plan: FlowCast

**What it is:** a traffic forecaster for 207 Los Angeles highway sensors. It predicts speed 15 to 60 minutes ahead, and a dashboard replays June 2012 so you can check every forecast against what actually happened.
**For:** ML and data people, and anyone who likes seeing a model graded in public. It shows forecasts beside the real outcome, so nothing has to be taken on trust.
**What sets it apart:** it is honest. Every forecast is drawn against reality, six models are scored on the same rows, and the README says where its own model *doesn't* win (it ties LightGBM at 60 min).
**Most impressive claim (from the README):** "Sensor 773939 is at 68 mph at 16:30. DCRNN forecasts 14 mph by 17:00. It was actually 22 mph." The 68 → 14 drop is the hook.
**Visual hook:** the network map, 207 dots turning orange and red as rush hour builds.
**Real UI to show:** Network (replay stepping 15:50 → 16:30), Sensor 773939 (the dashed forecast plunging, then what happened), Models (the error table).
**Tone:** `polished` with `default` energy. Data-serious, clean and confident. The app's own dark navy theme.
**Share caption:** "FlowCast forecasts LA freeway traffic up to an hour ahead, then shows you what actually happened."

## Angle
"Predict traffic before it happens," made concrete: ask a question at 16:30, then let the real app answer it, check itself against reality, and show its scorecard.

## Visual identity (from `frontend/src/styles/tokens.css`, dark theme)
- Ground `#0a111f`, surface `#111a2c`, border `#23314a`, text `#e8edf5`, muted `#a6b2c6`
- Forecast blue `#76a3da`; congestion ramp `#e0a458 → #e27d3e → #dd4f35 → #b42b20`
- Primary ivory `#ece7d9`; Inter Variable (from the built app's own font files), IBM Plex Mono for sensor IDs
- Logo: the app's own mark (the speed line with a dotted forecast tail), copied from `docs/assets/flowcast-hero-dark.svg`

## Source
The built static demo comes from the repo's `gh-pages` branch, served locally at `/flowcast/`. Scenes 3–5 are the **live app** in an iframe, with no screenshots. The camera, captions and highlights are layered on top, plus small injected styles (the chart reveal and the row highlight). Map tiles (OpenStreetMap) can't be reached from this machine, so the map shows sensors on the dark ground. That's fine for a launch video. No personal data appears anywhere: sensor IDs are from the public METR-LA dataset.

## Music
120 BPM (beat = 0.5 s, bar = 2 s), A minor, 22 s = 11 bars. Synthesized from scratch: soft kick, offbeat hats, sub bass, pad chords Am–F–C–G, and a light pluck arpeggio from scene 3 on. The cut SFX is a filtered whoosh on the grid, with ticks pitched to A for the replay steps. Every cut lands on a downbeat.

## Storyboard (22.0 s, 1920×1080, 30 fps)

| # | Time | Scene | On screen | Motion / sound |
|---|---|---|---|---|
| 1 | 0.0–4.0 | **Hook** | "Los Angeles. 16:30." then "Which freeways jam by 17:00?" Faint sensor dots behind (real coordinates from the app's data), drifting from grey to orange | Line 1 in at 0.25 s, line 2 in at 1.25 s, held to 4.0 s (5 words need 1.5 s). Pad and kick start, tick on each line |
| 2 | 4.0–7.0 | **Reveal** | FlowCast mark + "FLOWCAST", "Predict traffic before it happens." and "207 sensors · 15 to 60 minutes ahead" | Mark scales in, tagline rises, subline 0.5 s later. Whoosh in |
| 3 | 7.0–12.0 | **Highlight 1: the replay** | Real Network screen replaying 15:50 → 16:30, one 5-min step per beat (8 steps). The map recolours, the headline counts up and lands on "82 sensors forecast congested by 17:00". Caption: "Replay any moment. Every forecast on the map." | Slow push-in on the map; a tick per step |
| 4 | 12.0–16.5 | **Highlight 2: forecast vs reality** | Real Sensor 773939 page. KPI strip "68 mph → 14 mph", then the chart: history drawn, the dashed forecast and "what happened" wiped in to the right of Now. Caption: "Forecast 14 mph. It was 22." | Camera from the KPIs down to the chart; wipe on the beat |
| 5 | 16.5–19.5 | **Highlight 3: the scorecard** | Real Models table, DCRNN row highlighted. Caption: "Six models, same rows. Graded in public." | Push-in to the table, highlight sweeps in on the beat |
| 6 | 19.5–22.0 | **Outro** | Mark + "FlowCast", "See every forecast against what happened." and "lloydaq.github.io/flowcast · open source" | Settled hold, final chord rings out |

Transitions dip through the ground colour (0.2 s out, 0.2 s in), centred on each cut.
Reading-time check: the caption in scene 3 is 7 words (2.1 s) and is on screen 9.0–12.0. Scene 4's caption is 6 words, on 13.3–16.5. Scene 5's is 6 words, on 17.0–19.5. The outro text has 2.2 s of hold.
