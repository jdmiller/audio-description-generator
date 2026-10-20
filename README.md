# Audio description generator

Drafts WebVTT audio descriptions for a video by finding the moments where something changes on screen, then asking a local vision model to describe only what the captions don't already say. The output is a first draft for a human to review and edit. It does not replace that review.

## Why

Audio description is a WCAG requirement for prerecorded video (SC 1.2.5, AA), and writing it by hand is slow. For a large video library, an editable draft is a much better starting point than a blank page.

## How it works

`describer.py` processes each `.mp4` (from a local folder or an S3 prefix):

1. **Read the captions.** It loads the video's `.vtt` captions. Description goes only where nobody is speaking, so captions define the silent gaps.
2. **Find key moments.** [PySceneDetect](https://www.scenedetect.com/)'s `ContentDetector` flags cuts and large visual changes. `--threshold` sets sensitivity (default 7.0; lower finds more).
3. **Turn moments into gaps.** A moment becomes a description task only if it falls outside a caption cue and is at least 5 seconds after the previous task. Each window ends at the next caption, the next scene change, or 8 seconds, whichever is first.
4. **Write a premise.** The model first reads the whole transcript and writes a short summary of what the program is about, which it uses as context.
5. **Describe each gap.** For each window it samples three frames (start, middle, near the end) with ffmpeg and sends them to the model with the premise, the recent and upcoming dialogue, and its own last few descriptions. The prompt asks for objective, present-tense description with no mood, intent or poetic language. It tells the model to read on-screen text as it appears, describe charts by form before trend, and skip the gap if the dialogue already covers it or nothing new is visible.
6. **Remove repeats and write output.** Skipped or redundant results are dropped. The rest becomes `<video>_AD.vtt`, and a CSV report interleaves captions with descriptions for review.

The model runs locally through [LM Studio](https://lmstudio.ai/)'s OpenAI-compatible server at `http://localhost:1234/v1`. It is auto-detected, or set with `--model`. Load a vision-capable model.

With `--upload`, it can also publish the descriptions plus a self-hosted AblePlayer page to S3 and invalidate a CloudFront cache.

## Usage

```bash
brew install ffmpeg
pip install -r requirements.txt
# Start LM Studio's local server with a vision model loaded, then:
python3 describer.py --local ./videos             # .mp4 files with matching .vtt captions
python3 describer.py --local ./videos --test 1    # just the first video
python3 describer.py s3://my-bucket/videos/course-01 --output ./results
```

`run.sh` wraps the script to silence macOS malloc noise. For S3 mode, copy `config.example.json` to `config.json` (gitignored). Standard AWS environment variables or `~/.aws` credentials also work and take priority.

## Try the output in a player

`demo/` has a small page that plays a video with captions and text descriptions in [AblePlayer](https://ableplayer.github.io/ableplayer/). AblePlayer reads description cues aloud with the browser's text-to-speech and pauses the video while they play, so no recorded voice track is needed and the description text stays editable.

1. Download an [AblePlayer 4.8.0 release](https://github.com/ableplayer/ableplayer/releases) and copy `ableplayer.min.js`, `ableplayer.min.css`, `button-icons/`, `styles/` and `translations/` into `demo/`.
2. Add your own `video.mp4`, `captions.vtt` and the generated `descriptions.vtt` to `demo/`.
3. Serve the folder (`python3 -m http.server 8000`) and open `http://localhost:8000/demo/`.

To self-host the player in the `--upload` step, put the same AblePlayer files in `player/`.

## Limits

- Output is a draft. Models can describe things wrongly or miss them, and the prompt's style rules are instructions, not guarantees. Review every description before publishing.
- Descriptions are placed only in silent gaps, so dense dialogue gets few or none. Extended description (pausing the video) is not generated.
- Each description is given a fixed 4-second cue; the player's auto-pause decides how long it actually plays.
- Quality depends heavily on the model you load.

## Licence

MIT. See [LICENSE](LICENSE). AblePlayer (MIT), PySceneDetect (BSD-3) and other dependencies keep their own licences.
