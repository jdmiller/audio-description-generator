import os
os.environ.setdefault('MallocStackLogging', '0')  # suppress macOS noise before native libs load
import shutil
import csv
import re
import json
import logging
import warnings
import webvtt
import subprocess
import base64
import argparse
import tempfile
import sys
import datetime
import difflib
from urllib.parse import urlparse
from tqdm import tqdm
from openai import OpenAI

# --- PYTHON 3.13 COMPATIBILITY ---
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop
        sys.modules['audioop'] = audioop
    except ImportError:
        pass

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# --- CONFIG ---

def load_config():
    """Loads AWS credentials and CloudFront URL from config.json."""
    if not os.path.exists(CONFIG_FILE):
        print(f"⚠️  Warning: config.json not found at {CONFIG_FILE}. AWS calls may fail.")
        return {}
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

# --- HELPERS ---

def get_active_model(client):
    """Auto-detects the model currently loaded in LM Studio."""
    try:
        models = client.models.list()
        return models.data[0].id if models.data else None
    except Exception: return None

def format_vtt_time(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds); h = s // 3600; m = (s % 3600) // 60; s = s % 60
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def clean_ai_response(text):
    """Deep cleans AI output to remove JSON artifacts and meta-talk."""
    text = re.sub(r"```json|```", "", text).strip()
    text = re.sub(r"^(Here's|Based on|The image|I see|Description).*?:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace('{', '').replace('}', '').replace('"description":', '').replace('"', '').strip()
    return text[0].upper() + text[1:] if text else ""

def is_redundant(text, prev_text, recent_caps, threshold=0.85):
    """Programmatic failsafe to catch logically equivalent descriptions."""
    if not text: return True

    if prev_text and prev_text != "None":
        prev_lower = prev_text.strip().lower()
        text_lower_full = text.strip().lower()

        # Similarity check (catches rephrasing of the same content)
        sim = difflib.SequenceMatcher(None, text_lower_full, prev_lower).ratio()
        if sim > threshold: return True

        # Superset check: new description CONTAINS the entire previous one.
        # Pattern: model outputs prev_desc + new_stuff. Only new_stuff is needed,
        # but we can’t split it cleanly — mark as redundant so the model re-runs
        # with a stronger prompt next time (or the new info is already in the stream).
        if len(prev_lower) > 30 and prev_lower in text_lower_full:
            return True

        # Prefix check: new description starts with the first 60 chars of prev.
        # Catches cases where the model mostly repeats then extends.
        if len(prev_lower) > 30 and text_lower_full.startswith(prev_lower[:60]):
            return True

    text_lower = text.lower()
    for cap in recent_caps:
        if text_lower in cap.lower() or cap.lower() in text_lower:
            return True
        sim = difflib.SequenceMatcher(None, text_lower, cap.lower()).ratio()
        if sim > threshold: return True

    return False

def get_local_captions(video_path):
    """Loads captions from a local .vtt or .srt file alongside the video."""
    base = os.path.splitext(video_path)[0]
    for ext in ['.vtt', '.srt']:
        path = f"{base}{ext}"
        if os.path.exists(path):
            try:
                caps = webvtt.read(path) if ext == '.vtt' else webvtt.from_srt(path)
                return [{"start": c.start_in_seconds, "end": c.end_in_seconds, "text": c.text.strip()} for c in caps]
            except Exception: return []
    return []

# --- S3 HELPERS ---

def parse_s3_url(s3_url):
    """Parses an s3://bucket/prefix URL into (bucket, prefix)."""
    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URL: '{s3_url}'. Must start with s3://")
    return parsed.netloc, parsed.path.lstrip('/')

def get_s3_client(config):
    """
    Creates a boto3 S3 client.

    Credential resolution order (first non-empty value wins):
      1. Standard AWS env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
         AWS_SESSION_TOKEN, AWS_DEFAULT_REGION) — supports SSO, assumed roles,
         `aws sts get-session-token`, CI/CD injection, etc.
      2. config.json values as a fallback.

    Passing None for a value lets boto3 continue down its own default chain
    (~/.aws/credentials, EC2 instance profile, etc.).
    """
    import boto3

    key    = os.environ.get('AWS_ACCESS_KEY_ID')     or config.get('aws_access_key_id')     or None
    secret = os.environ.get('AWS_SECRET_ACCESS_KEY') or config.get('aws_secret_access_key') or None
    token  = os.environ.get('AWS_SESSION_TOKEN')     or config.get('aws_session_token')     or None
    region = os.environ.get('AWS_DEFAULT_REGION')    or config.get('aws_region', 'us-east-1')

    if token:
        print("  🔑 Using temporary credentials (session token present)")

    return boto3.client(
        's3',
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        aws_session_token=token,
        region_name=region,
    )

def list_s3_mp4s(s3_client, bucket, prefix):
    """Lists all .mp4 files under the given S3 bucket/prefix, skipping hidden files."""
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            basename = os.path.basename(key)
            if key.endswith('.mp4') and not basename.startswith('._'):
                keys.append(key)
    return sorted(keys)

def check_s3_caption_exists(s3_client, bucket, video_key):
    """Returns the S3 key of a companion .vtt or .srt caption file if it exists."""
    base = os.path.splitext(video_key)[0]
    for ext in ['.vtt', '.srt']:
        cap_key = f"{base}{ext}"
        try:
            s3_client.head_object(Bucket=bucket, Key=cap_key)
            return cap_key
        except Exception:
            continue
    return None

def get_s3_captions(s3_client, bucket, video_key):
    """Tries to fetch a companion .vtt or .srt caption file from S3 for a given video key."""
    base = os.path.splitext(video_key)[0]
    for ext in ['.vtt', '.srt']:
        cap_key = f"{base}{ext}"
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=cap_key)
            content = obj['Body'].read().decode('utf-8')
            # webvtt-py needs a real file, so write to a short-lived temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                caps = webvtt.read(tmp_path) if ext == '.vtt' else webvtt.from_srt(tmp_path)
                return [{"start": c.start_in_seconds, "end": c.end_in_seconds, "text": c.text.strip()} for c in caps]
            except Exception:
                return []
            finally:
                os.unlink(tmp_path)
        except Exception:
            continue  # no companion caption file found — that's fine
    return []

def upload_to_s3(s3_client, local_dir, bucket, s3_dest_prefix):
    """
    Recursively uploads a local directory to an S3 prefix, setting correct
    Content-Type headers for each file type so CloudFront serves them properly.
    """
    import mimetypes

    # Ensure webvtt and other types are mapped correctly
    _CONTENT_TYPES = {
        '.html': 'text/html; charset=utf-8',
        '.vtt':  'text/vtt; charset=utf-8',
        '.srt':  'text/plain; charset=utf-8',
        '.js':   'application/javascript',
        '.css':  'text/css',
        '.svg':  'image/svg+xml',
        '.png':  'image/png',
        '.jpg':  'image/jpeg',
        '.json': 'application/json',
    }

    dest_prefix = s3_dest_prefix.rstrip('/')
    uploaded = 0

    for root, _, files in os.walk(local_dir):
        for filename in files:
            if filename.startswith('._') or filename == '.DS_Store':
                continue
            local_path = os.path.join(root, filename)
            # Build the S3 key relative to local_dir
            rel_path   = os.path.relpath(local_path, local_dir)
            s3_key     = f"{dest_prefix}/{rel_path.replace(os.sep, '/')}"
            ext        = os.path.splitext(filename)[1].lower()
            content_type = _CONTENT_TYPES.get(ext, mimetypes.guess_type(filename)[0] or 'application/octet-stream')

            s3_client.upload_file(
                local_path, bucket, s3_key,
                ExtraArgs={'ContentType': content_type}
            )
            uploaded += 1

    return uploaded


def invalidate_cloudfront(config, s3_prefix):
    """
    Creates a CloudFront invalidation for all files under the audio-descriptions
    path, clearing any previously cached versions.
    Requires 'cloudfront_distribution_id' in config.json.
    """
    import boto3

    dist_id = config.get('cloudfront_distribution_id')
    if not dist_id:
        print("  ⚠️  Skipping CloudFront invalidation: 'cloudfront_distribution_id' not set in config.json")
        return

    key    = os.environ.get('AWS_ACCESS_KEY_ID')     or config.get('aws_access_key_id')     or None
    secret = os.environ.get('AWS_SECRET_ACCESS_KEY') or config.get('aws_secret_access_key') or None
    token  = os.environ.get('AWS_SESSION_TOKEN')     or config.get('aws_session_token')     or None
    region = os.environ.get('AWS_DEFAULT_REGION')    or config.get('aws_region', 'us-east-1')

    cf_client = boto3.client(
        'cloudfront',
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        aws_session_token=token,
        region_name=region,
    )

    invalidation_path = f"/{s3_prefix.strip('/')}/audio-descriptions/*"
    response = cf_client.create_invalidation(
        DistributionId=dist_id,
        InvalidationBatch={
            'Paths': {'Quantity': 1, 'Items': [invalidation_path]},
            'CallerReference': str(datetime.datetime.now().timestamp()),
        }
    )
    inv_id = response['Invalidation']['Id']
    print(f"  ✅ CloudFront invalidation created: {inv_id}")
    print(f"     Path: {invalidation_path}")


# --- HTML GENERATOR ---

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_player_dir(ad_dir):
    """
    Copies the local player/ folder (AblePlayer assets) into ad_dir/player/ so it
    gets uploaded to S3 alongside the VTTs via the single aws s3 cp command.
    Large piper model files (*.onnx) and the unminified source are excluded.
    """
    src = os.path.join(_SCRIPT_DIR, 'player')
    dst = os.path.join(ad_dir, 'player')
    if not os.path.isdir(src):
        print(f"  ⚠️  player/ directory not found at {src} — AblePlayer will not be self-hosted.")
        return dst
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst,
                    ignore=shutil.ignore_patterns('*.onnx', '*.onnx.json', 'ableplayer.js'))
    print(f"  📦 Player assets staged: {dst}")
    return dst

_HTML_CSS = """
:root {
    --primary: #005a9c;
    --primary-hover: #003d6b;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #333;
    --text-light: #666;
    --shadow: 0 4px 6px rgba(0,0,0,0.1);
    --radius: 12px;
}

*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.5;
    color: var(--text);
    background-color: var(--bg);
    margin: 0;
    padding: 0;
}

.container {
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

header {
    text-align: center;
    margin-bottom: 3rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid #e1e4e8;
}

h1 {
    color: var(--primary);
    font-weight: 800;
    font-size: 2.2rem;
    margin: 0 0 0.5rem;
}

.subtitle {
    color: var(--text-light);
    font-size: 1.05rem;
    margin: 0;
}

/* Collapsible video sections */
.video-section {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    border: 1px solid #e1e4e8;
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.video-section summary {
    list-style: none;
    cursor: pointer;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    user-select: none;
    transition: background 0.15s;
}

.video-section summary::-webkit-details-marker { display: none; }

.video-section summary:hover { background: #f0f4f8; }

.video-section summary:focus-visible {
    outline: 3px solid #ffbf47;
    outline-offset: -3px;
}

.summary-title { flex: 1; }

.summary-chevron {
    width: 1.5rem;
    height: 1.5rem;
    flex-shrink: 0;
    color: var(--primary);
    transition: transform 0.25s ease;
}

details[open] .summary-chevron {
    transform: rotate(180deg);
}

.player-wrapper {
    background: #000;
    border-top: 1px solid #e1e4e8;
}

/* Nudge AblePlayer to fill the card width */
.player-wrapper video { width: 100%; }

@media (max-width: 640px) {
    h1 { font-size: 1.6rem; }
    .container { padding: 1rem; }
    .video-section summary { font-size: 1rem; padding: 1rem; }
}
"""

def build_ableplayer_html(folder_name, s3_prefix, cloudfront_url, video_entries, player_url=None):
    """
    Builds the complete single-page HTML file with one collapsible AblePlayer per video.

    Args:
        folder_name:     The local output folder name (= last segment of S3 prefix).
        s3_prefix:       The full S3 prefix path, e.g. "videos/course-01".
        cloudfront_url:  CloudFront base URL (no trailing slash).
        video_entries:   List of dicts with keys: display_name, video_key,
                         caption_key (may be None), ad_vtt_name.
        player_url:      CloudFront URL to the self-hosted player/ directory.
                         e.g. "https://xxx.cloudfront.net/videos/course-01/audio-descriptions/player"
    """
    cf     = cloudfront_url.rstrip('/')
    prefix = s3_prefix.strip('/')
    p      = (player_url or '').rstrip('/')

    sections = []
    for idx, entry in enumerate(video_entries, start=1):
        display_name  = entry['display_name']
        video_key     = entry['video_key']       # full S3 key
        caption_key   = entry.get('caption_key') # full S3 key or None
        ad_vtt_name   = entry['ad_vtt_name']     # just the filename, e.g. "lecture1_AD.vtt"

        video_url   = f"{cf}/{video_key}"
        # AD VTTs are uploaded to an audio-descriptions/ subfolder in that S3 prefix
        ad_url      = f"{cf}/{prefix}/audio-descriptions/{ad_vtt_name}"
        caption_tag = ""
        if caption_key:
            cap_url     = f"{cf}/{caption_key}"
            caption_tag = f'\n                    <track kind="captions" src="{cap_url}" srclang="en" label="English Captions" default>'

        player_id = f"video-{idx}"

        sections.append(f"""
        <div class="video-section-wrapper">
          <details class="video-section">
            <summary>
              <span class="summary-title">{display_name}</span>
              <svg class="summary-chevron" aria-hidden="true" focusable="false"
                   xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </summary>
            <div class="player-wrapper">
              <video id="{player_id}"
                     data-able-player
                     data-desc-pause-default="on"
                     data-state-descriptions="on"
                     crossorigin="anonymous"
                     preload="none"
                     width="100%">{caption_tag}
                <track kind="descriptions" src="{ad_url}" srclang="en" label="Audio Descriptions">
                <source src="{video_url}" type="video/mp4">
              </video>
            </div>
          </details>
        </div>""")

    sections_html = "\n".join(sections)
    title         = folder_name.replace('-', ' ').replace('_', ' ').title()

    # Use self-hosted assets when player_url is available, otherwise fall back to CDN.
    # jQuery must always be loaded first — ableplayer.dist.js does NOT bundle it.
    jquery_url = 'https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js'
    if p:
        player_css    = f'{p}/ableplayer.min.css'
        player_js     = f'{p}/ableplayer.dist.js'
        player_config = f'<script>window.AblePlayerConfig = {{ pathToPlayer: "{p}/" }};</script>'
    else:
        player_css    = 'https://cdn.jsdelivr.net/npm/ableplayer@4.6.1/build/ableplayer.min.css'
        player_js     = 'https://cdn.jsdelivr.net/npm/ableplayer@4.6.1/build/ableplayer.min.js'
        player_config = ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} – Video Library</title>
  <meta name="description" content="Accessible video library for {title} with audio descriptions.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{player_css}">
  <script src="{jquery_url}"></script>
  {player_config}
  <script src="{player_js}"></script>
  <style>{_HTML_CSS}</style>
</head>
<body>
  <div class="container">
    <header role="banner">
      <h1>{title}</h1>
      <p class="subtitle">Select a video to watch with full accessibility features including audio descriptions.</p>
    </header>
    <main>
{sections_html}
    </main>
  </div>
</body>
</html>
"""

# --- ERROR LOGGING ---

def setup_error_logging(output_dir, ts):
    """
    Configures a run-scoped error log file and silences all third-party library
    noise so the console shows only our explicit print() output.

    Everything that would normally leak to stderr or Python's warning system
    (scenedetect progress, boto3 retries, urllib3 warnings, etc.) is routed
    to errors_TIMESTAMP.log instead.

    Returns the path to the log file.
    """
    log_path = os.path.join(output_dir, f"errors_{ts}.log")

    # File handler: DEBUG and above → log file (third-party loggers are
    # silenced separately, so only our own logging.info/debug calls get through)
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'
    ))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)

    # Route Python warnings to the log file instead of stderr
    logging.captureWarnings(True)
    warnings.filterwarnings('default')

    # Silence verbose third-party loggers entirely
    for lib in ('scenedetect', 'pyscenedetect', 'httpx', 'httpcore', 'openai', 'urllib3',
                'boto3', 'botocore', 's3transfer', 'webvtt'):
        logging.getLogger(lib).setLevel(logging.CRITICAL)

    # Redirect sys.stderr so any remaining library output goes to the log
    class _FileStderr:
        def __init__(self, path):
            self._f = open(path, 'a', encoding='utf-8')
        def write(self, s):
            if s and s.strip():
                self._f.write(s)
                self._f.flush()
        def flush(self):
            self._f.flush()
        def fileno(self):
            return self._f.fileno()

    sys.stderr = _FileStderr(log_path)

    # Ensure uncaught exceptions also land in the log
    _orig_hook = sys.__excepthook__
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        logging.critical('Uncaught exception:\n' + ''.join(
            traceback.format_exception(exc_type, exc_value, exc_tb)
        ))
    sys.excepthook = _excepthook

    return log_path


# --- NARRATIVE WORKER ---

def generate_program_premise(caps, hint, c_params, model):
    """
    Before processing any frames, ask the LLM to read the full transcript and
    produce a 2-3 sentence 'program premise' — what this content is about, who
    appears, what subject matter is covered. This anchors every subsequent
    description to the right domain (e.g. wildfire documentary vs geology lecture).
    Returns an empty string if caps is empty or the call fails.
    """
    if not caps:
        return ""

    full_text = " ".join(c["text"] for c in caps)
    if not full_text.strip():
        return ""

    client = OpenAI(base_url=c_params['url'], api_key=c_params['key'])
    prompt = (
        f"The following is the full spoken transcript of a video titled '{hint}'.\n"
        f"TRANSCRIPT:\n{full_text}\n\n"
        "In 2-3 sentences, summarise: (1) what this program is about, (2) any named people, "
        "places, or organisations that appear, and (3) the subject domain (e.g. science, "
        "news, training, nature). Be factual and concise. This will be used as context for "
        "a professional audio describer."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"generate_program_premise failed for '{hint}': {e}")
        return ""

def ai_worker(start_s, end_s, video_path, caps, hint, prev_descs, args, c_params,
              premise="", forward_caps=None):
    """
    Analyzes a 3-frame sequence spanning the full silent gap [start_s, end_s].

    Args:
        prev_descs:   Rolling list of the last N emitted descriptions (not just one).
        premise:      Program-level summary generated from the full transcript.
        forward_caps: Captions that begin AFTER this gap — a lookahead clue about
                      what the visuals are illustrating.
    """
    client = OpenAI(base_url=c_params['url'], api_key=c_params['key'])

    # 15s context window of dialogue immediately before the gap
    recent = [c["text"] for c in caps if (start_s - 15) <= c["end"] <= start_s]
    ctx = " ".join(recent) if recent else "Silence"

    # Forward lookahead: next 2 caption cues after the gap
    ahead = " ".join((forward_caps or [])[:2])

    # Rolling narrative: last N descriptions (most recent last)
    narrative = " → ".join(prev_descs) if prev_descs else "None"
    # For deduplication the most recent description is the last item
    prev_desc = prev_descs[-1] if prev_descs else "None"

    # Sample 3 frames: start, midpoint, near-end of the gap
    gap     = max(end_s - start_s, 0.5)
    f_times = [
        start_s,
        start_s + gap * 0.5,
        end_s - min(0.25, gap * 0.1),
    ]

    # Build the context header — richer when premise and lookahead are available
    context_block = f"PROGRAMME CONTEXT: {premise}\n" if premise else ""
    context_block += f"RECENT DIALOGUE (before gap): \"{ctx}\"\n"
    context_block += f"UPCOMING DIALOGUE (after gap): \"{ahead}\"\n" if ahead else ""
    context_block += f"NARRATIVE SO FAR: {narrative}\n"
    context_block += f"YOUR LAST DESCRIPTION: \"{prev_desc}\"\n"

    prompt = (
        f"You are a professional audio describer for '{hint}'.\n"
        f"{context_block}\n"
        f"You are seeing 3 frames sampled across a {gap:.1f}-second silent gap "
        "(beginning, middle, near-end). Use ALL context above — especially the "
        "upcoming dialogue — to determine what is being depicted before describing it.\n\n"
        "RULES:\n"
        "1. DESCRIBE ACTION AND VISUALS: Prioritise what is HAPPENING or CHANGING across "
        "the frames — actions, people, objects, on-screen text, spatial relationships. "
        "If something moves, transforms, or progresses, describe that arc as a unified event "
        "(e.g. 'A firefighter carries equipment toward a burning building'). "
        "Never describe silence, sound, or mood.\n"
        "2. DEDUPLICATE — DIALOGUE: If the dialogue already names the action or object, set 'skip' to true.\n"
        "3. NO REPETITION — PREVIOUS DESCRIPTION: Do NOT repeat content from your previous description. "
        "Your response must describe only what is NEW or DIFFERENT from the last description. "
        "If any element (logo, background, object) was described previously, do not mention it again "
        "unless it has visibly changed. Only set 'skip' to true if the scene is "
        "essentially identical to your last description — same subject, same action, same framing, "
        "with nothing meaningfully new visible.\n"
        "4. UNIFIED SEQUENCE: Describe the 3-frame sequence as one coherent event, not frame-by-frame.\n"
        "5. OBJECTIVE: Use present tense, active voice, third-person. No interpretation, emotion, "
        "inferred meaning, or implied cause (e.g., not 'hinting at danger,' 'a silent threat looms,' "
        "'anticipating,' 'indicating water flow,' 'suggesting erosion,' 'implying danger'). "
        "Describe only what is directly visible — not what it might mean or cause.\n"
        "6. NO POETIC LANGUAGE: Avoid words like 'profound,' 'vast,' 'relentless,' 'shrouded,' "
        "'enveloping,' 'looms,' 'descends.' Describe shape, size, color, and movement instead.\n"
        "7. PEOPLE AND OBJECTS: Identify by name if known, or by a visible, functionally useful "
        "attribute (e.g., 'a man in a red jacket'). Describe actions and posture only — never emotion "
        "or intent. Do not include race or ethnicity. If you cannot identify an object with confidence, "
        "describe its observable properties — shape, size, color, how it is held — without guessing "
        "its name (e.g., not 'a device resembling a grenade launcher' — instead 'a man holds a "
        "shoulder-mounted tube, pointed toward the slope').\n"
        "8. ON-SCREEN TEXT AND LOGOS: Read text as it appears. Do not interpret what it signifies "
        "or represents (e.g., not 'signifying wilderness' — just describe the logo and its text).\n"
        "9. DATA VISUALIZATIONS: Describe the visual form first (axes, colors, shape of data), "
        "then the trend if it is clearly labeled. Do not state conclusions the viewer should draw themselves.\n"
        "10. BLACK SCREEN / UNRESOLVABLE FRAME: If all frames are black or static with no content, "
        "output exactly: 'The screen is black.' If the frame cannot be analyzed clearly, "
        "set 'skip' to true rather than returning an error.\n"
        "11. SUCCINCTNESS: Target under 15 words. Max 40 for complex multi-step actions or scenes "
        "with multiple elements. One description per gap — do not combine unrelated elements from different cuts.\n"
        "12. NO META-TALK: Begin directly with the subject (e.g., 'Floodwater covers a street...').\n"
        "\nRESPONSE (JSON): {\"description\": \"objective present-tense sentence\", \"skip\": false}"
    )

    try:
        # Extract frames via ffmpeg, skipping any that return empty bytes
        # (e.g. seek past EOF on a short gap)
        frames_b64 = []
        for ft in f_times:
            f_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
                     '-ss', str(ft), '-i', video_path,
                     '-frames:v', '1', '-q:v', '2', '-pix_fmt', 'yuvj420p', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-']
            try:
                raw_bytes = subprocess.check_output(f_cmd, stderr=subprocess.DEVNULL)
                if raw_bytes:  # skip empty output (seek past EOF)
                    frames_b64.append(base64.b64encode(raw_bytes).decode('utf-8'))
            except subprocess.CalledProcessError:
                pass  # this frame failed; continue with whatever we have

        if not frames_b64:
            raise ValueError(f"No valid frames could be extracted from [{start_s:.1f}s, {end_s:.1f}s]")

        # Build content list: text prompt + valid frames only
        content = [{"type": "text", "text": prompt}]
        for img in frames_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}]
        )

        raw_content = resp.choices[0].message.content
        logging.debug(f"[{hint}] {start_s:.1f}s model output: {raw_content!r}")

        # Normalise: strip any markdown code fences
        clean = re.sub(r'```(?:json)?\s*', '', raw_content).strip()

        data = None

        # Strategy 1: slice first-'{' to last-'}' then json.loads.
        # Handles any amount of prose before or after the JSON object
        # (the most common failure mode: "Here's the description: {...}").
        try:
            first = clean.index('{')
            last  = clean.rindex('}')
            data  = json.loads(clean[first:last + 1])
        except (ValueError, json.JSONDecodeError):
            pass

        # Strategy 2: raw_decode — better when model outputs multiple JSON objects 
        # concatenated; stops cleanly after the first complete one.
        if data is None:
            try:
                first = clean.index('{')
                data, _ = json.JSONDecoder().raw_decode(clean, first)
            except (ValueError, json.JSONDecodeError):
                pass

        # Strategy 3: regex field extraction — last resort for truncated responses.
        if data is None:
            m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
            if m:
                data = {"description": m.group(1), "skip": False}

        if data is None:
            raise AttributeError("no valid JSON in model output")

        text = clean_ai_response(data.get("description", ""))

        # Apply programmatic deduplication failsafe alongside AI's own skip flag
        ai_skip   = data.get("skip", False)
        prog_skip = is_redundant(text, prev_desc, recent)

        return {"time": start_s, "text": text, "skip": ai_skip or prog_skip, "error": None}

    except subprocess.CalledProcessError as e:
        msg = f"ffmpeg failed at {start_s:.1f}s (returncode={e.returncode})"
        logging.error(msg)
        tqdm.write(f"    ⚠️  ffmpeg error at {start_s:.1f}s — see error log")
        return {"time": start_s, "text": "Analysis Error", "skip": True, "error": msg}
    except AttributeError:
        raw = resp.choices[0].message.content[:300] if resp else "no response"
        msg = f"JSON parse failed at {start_s:.1f}s. Model output: {raw!r}"
        logging.error(msg)
        tqdm.write(f"    ⚠️  JSON parse error at {start_s:.1f}s — see error log")
        return {"time": start_s, "text": "Analysis Error", "skip": True, "error": msg}
    except Exception as e:
        msg = f"{type(e).__name__} at {start_s:.1f}s: {e}"
        logging.error(msg)
        tqdm.write(f"    ⚠️  Error at {start_s:.1f}s — see error log")
        return {"time": start_s, "text": "Analysis Error", "skip": True, "error": msg}

# --- PROCESSOR ---

def process_video(v_path, args, c_params, report_path, output_dir=None, caps=None, display_name=None):
    """
    Processes a single video file and returns the path of the generated _AD.vtt file.

    Args:
        v_path:       Local filesystem path to the video (may be a temp file for S3 sources).
        output_dir:   Where to save the output VTT file. Defaults to the video's own directory.
        caps:         Pre-loaded captions list (supplied by S3 mode). If None, loads from disk.
        display_name: Human-readable name (e.g. the original S3 key basename without extension).

    Returns:
        str: Absolute path of the saved _AD.vtt file.
    """
    from scenedetect import detect, ContentDetector

    v_name = display_name if display_name else os.path.basename(v_path).rsplit('.', 1)[0]
    out_dir = output_dir if output_dir else os.path.dirname(v_path)
    print(f"\n🎬 Processing: {v_name}")

    if caps is None:
        caps = get_local_captions(v_path)

    # Suppress scenedetect's own stderr output — errors surface via exception
    import contextlib, io
    with contextlib.redirect_stderr(io.StringIO()) as _sd_stderr:
        scene_list = detect(v_path, ContentDetector(threshold=args.threshold))
    _sd_out = _sd_stderr.getvalue()
    if _sd_out.strip():
        logging.warning(f"scenedetect stderr for '{v_name}': {_sd_out.strip()}")
    triggers = sorted(list(set([s[0].get_seconds() for s in scene_list])))
    if not triggers or triggers[0] > 1.0: triggers.insert(0, 0.0)

    # Build (start, end) task windows for each silent gap.
    # end_s = the earlier of: next caption start, next trigger, or start + 8s.
    # This gives the AI the full arc of each scene gap to reason about.
    tasks, last_t, MIN_GAP, MAX_WINDOW = [], -999.0, 5.0, 8.0
    for i, t in enumerate(triggers):
        in_cap = any(c["start"] <= t <= c["end"] for c in caps)
        if not in_cap and t >= last_t + MIN_GAP:
            next_trigger  = triggers[i + 1] if i + 1 < len(triggers) else None
            next_cap      = next((c["start"] for c in caps if c["start"] > t), None)
            candidates    = [x for x in [next_trigger, next_cap] if x is not None]
            end_s         = min(min(candidates) if candidates else t + MAX_WINDOW, t + MAX_WINDOW)
            tasks.append((t, end_s))
            last_t = t

    logging.info(f"[{v_name}] START — {len(caps)} caption cues loaded")
    logging.info(f"[{v_name}] Scene detection: {len(triggers)} triggers → {len(tasks)} silent-gap tasks")

    # ── Step 1: generate a program premise from the full transcript ───────────
    if caps:
        print("  🔍 Generating program premise from transcript ...")
        premise = generate_program_premise(caps, v_name, c_params, args.model)
        if premise:
            print(f"  📋 Premise: {premise[:120]}{'...' if len(premise) > 120 else ''}")
            logging.info(f"[{v_name}] PREMISE: {premise}")
        else:
            logging.info(f"[{v_name}] PREMISE: (none generated — transcript may be empty)")
    else:
        premise = ""
        logging.info(f"[{v_name}] PREMISE: (skipped — no captions)")

    # Sequential processing to maintain rolling context for deduplication
    ROLLING_WINDOW = 4   # number of past descriptions to keep in narrative context
    raw_ad, emitted_descs = [], []
    for (t, end_s) in tqdm(tasks, desc="🧠 Analysis", colour="cyan",
                            position=1, leave=False, dynamic_ncols=True,
                            file=sys.stdout):
        # Forward lookahead: next 2 captions that begin after this gap
        forward_caps = [c["text"] for c in caps if c["start"] > end_s][:2]

        res = ai_worker(
            t, end_s, v_path, caps, v_name,
            prev_descs   = emitted_descs[-ROLLING_WINDOW:],
            args         = args,
            c_params     = c_params,
            premise      = premise,
            forward_caps = forward_caps,
        )
        if res.get("error"):
            logging.info(f"[{v_name}] {t:.1f}s — ERROR: {res['error']}")
        elif res["skip"]:
            logging.info(f"[{v_name}] {t:.1f}s — SKIP: {res['text']!r}")
        else:
            logging.info(f"[{v_name}] {t:.1f}s — KEEP: {res['text']!r}")
            emitted_descs.append(res["text"])
        raw_ad.append(res)

    # Assemble rows for the Master Report (KEEP and ERROR only — REMOVE excluded)
    video_rows = []
    for c in caps:
        video_rows.append({
            "video name": v_name,
            "timestamp":  format_vtt_time(c["start"]),
            "content":    c["text"],
        })
    for d in raw_ad:
        if d["skip"] and not d.get("error"):
            continue  # REMOVE — logged to file, excluded from CSV
        content = d.get("error") or d["text"]
        video_rows.append({
            "video name": v_name,
            "timestamp":  format_vtt_time(d["time"]),
            "content":    f"[Audio Description] {content}",
        })

    video_rows.sort(key=lambda x: x['timestamp'])

    # Incremental write to Master Report
    with open(report_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["video name", "timestamp", "content"])
        writer.writerows(video_rows)

    # Save VTT to a _processing temp name first; rename atomically on success
    vtt = webvtt.WebVTT()
    for d in sorted(raw_ad, key=lambda x: x['time']):
        # Exclude: intentional deduplication skips, errors, and any stray short strings
        if not d["skip"] and not d.get("error") and len(d["text"]) > 5:
            vtt.captions.append(webvtt.Caption(
                format_vtt_time(d["time"]),
                format_vtt_time(d["time"] + 4.0),
                d["text"]
            ))

    vtt_processing = os.path.join(out_dir, f"{v_name}_AD_processing.vtt")
    vtt_final      = os.path.join(out_dir, f"{v_name}_AD.vtt")
    vtt.save(vtt_processing)
    os.replace(vtt_processing, vtt_final)   # atomic rename — only visible when complete
    kept   = sum(1 for d in raw_ad if not d["skip"] and not d.get("error"))
    removed = sum(1 for d in raw_ad if d["skip"] and not d.get("error"))
    errors  = sum(1 for d in raw_ad if d.get("error"))
    logging.info(
        f"[{v_name}] DONE — {kept} KEEP / {removed} REMOVE / {errors} ERROR "
        f"out of {len(raw_ad)} tasks. VTT: {vtt_final}"
    )
    print(f"  ✅ VTT saved: {vtt_final}")
    return vtt_final

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(
        description="Audio description generator. Defaults to S3 mode; use --local for local directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  S3 mode:    python describer.py s3://my-bucket/videos/course-01\n"
            "  Local mode: python describer.py --local ./videos\n"
            "  S3 + out:   python describer.py s3://my-bucket/videos --output ./results"
        )
    )
    parser.add_argument("source", nargs='?', help="S3 URL to process (e.g. s3://bucket/prefix)")
    parser.add_argument("--local", metavar="DIR", help="Process a local directory of .mp4 files instead of S3")
    parser.add_argument("--output", metavar="DIR", help="Output directory for VTT files and report (S3 mode). Defaults to current directory.")
    parser.add_argument("--model", default=None, help="LM Studio model ID (auto-detected if omitted)")
    parser.add_argument("--threshold", type=float, default=7.0, help="Scene detection sensitivity (default: 7.0 — lower = more scenes detected)")
    parser.add_argument("--upload", action="store_true",
                        help="Skip processing — upload existing audio-descriptions/ folder to S3 and invalidate CloudFront")
    parser.add_argument("--test", nargs='?', type=int, const=1, default=None,
                        metavar="N", help="Process only the first N videos (default: 1 if flag is used)")
    args = parser.parse_args()

    # Redirect fd 2 (C-level stderr) to /dev/null as early as possible so all
    # child processes (scenedetect multiprocessing workers, etc.) inherit a
    # silent stderr and can't print MallocStackLogging noise to the terminal.
    _devnull_fd = os.open('/dev/null', os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)

    # --- Validate source arguments ---
    if args.local and args.source:
        parser.error("Specify either a positional S3 URL or --local DIR, not both.")
    if not args.local and not args.source:
        parser.error("You must provide an S3 URL (e.g. s3://bucket/prefix) or use --local DIR.")

    # --- LM Studio client setup (skipped in --upload mode) ---
    client_params = {'url': "http://localhost:1234/v1", 'key': "lm-studio"}
    if not args.upload:
        client = OpenAI(base_url=client_params['url'], api_key=client_params['key'])
        if not args.model:
            try:
                models = client.models.list()
                args.model = models.data[0].id if models.data else None
            except Exception: pass
        if not args.model:
            print("❌ Error: No model loaded in LM Studio.")
            return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------------------------------------------------------------------------
    # LOCAL MODE
    # -------------------------------------------------------------------------
    if args.local:
        if not os.path.isdir(args.local):
            print(f"❌ Error: Local directory not found: {args.local}")
            return

        report_path = os.path.join(args.local, f"ADReport_{ts}.csv")
        error_log   = setup_error_logging(args.local, ts)
        print(f"📋 Error log: {error_log}")
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["video name", "timestamp", "content"])
            writer.writeheader()

        v_files = sorted([
            os.path.join(args.local, f)
            for f in os.listdir(args.local)
            if f.endswith('.mp4') and not f.startswith('._')
        ])
        if not v_files:
            print(f"❌ No .mp4 files found in {args.local}")
            return

        if args.test is not None:
            v_files = v_files[:args.test]
            print(f"🧪 Test mode — limiting to {len(v_files)} video(s)")

        print(f"📁 Found {len(v_files)} video(s) in {args.local}")
        for v in v_files:
            process_video(v, args, client_params, report_path)

        print(f"\n📊 Master Report saved: {report_path}")
        return

    # -------------------------------------------------------------------------
    # S3 MODE
    # -------------------------------------------------------------------------
    config    = load_config()
    s3        = get_s3_client(config)
    bucket, s3_prefix = parse_s3_url(args.source)
    cloudfront_url    = config.get('cloudfront_distribution_url', '').rstrip('/')

    if not cloudfront_url:
        print("⚠️  Warning: 'cloudfront_distribution_url' not set in config.json. HTML URLs will be incomplete.")

    # Local output folder named after the last segment of the S3 prefix
    folder_name = s3_prefix.rstrip('/').split('/')[-1] or bucket
    output_root = args.output if args.output else os.getcwd()
    output_dir  = os.path.join(output_root, folder_name)
    ad_dir      = os.path.join(output_dir, "audio-descriptions")
    os.makedirs(ad_dir, exist_ok=True)

    # ── Upload-only mode: skip all processing ─────────────────────────────────
    if args.upload:
        if not os.path.isdir(ad_dir) or not os.listdir(ad_dir):
            print(f"❌ No audio-descriptions folder (or it is empty) at:\n   {ad_dir}")
            print("    Run without --upload first to generate VTT files.")
            return
        ad_s3_prefix = f"{s3_prefix.strip('/')}/audio-descriptions"
        print(f"⬆️  Uploading {ad_dir}")
        print(f"    → s3://{bucket}/{ad_s3_prefix} ...")
        try:
            n = upload_to_s3(s3, ad_dir, bucket, ad_s3_prefix)
            print(f"  ✅ Uploaded {n} file(s)")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            return
        print(f"\n🔄 Invalidating CloudFront cache ...")
        try:
            invalidate_cloudfront(config, s3_prefix)
        except Exception as e:
            print(f"  ❌ CloudFront invalidation failed: {e}")
        return
    # ───────────────────────────────────────────────────────────────────────────────

    report_path = os.path.join(output_dir, f"ADReport_{ts}.csv")
    error_log   = setup_error_logging(output_dir, ts)
    print(f"📋 Error log: {error_log}")
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["video name", "timestamp", "content"])
        writer.writeheader()

    if cloudfront_url:
        print(f"☁️  CloudFront: {cloudfront_url}")

    mp4_keys = list_s3_mp4s(s3, bucket, s3_prefix)
    if not mp4_keys:
        print(f"❌ No .mp4 files found at s3://{bucket}/{s3_prefix}")
        return

    if args.test is not None:
        mp4_keys = mp4_keys[:args.test]
        print(f"🧪 Test mode — limiting to {len(mp4_keys)} video(s)")

    print(f"📦 Found {len(mp4_keys)} video(s) in s3://{bucket}/{s3_prefix}")
    print(f"📂 Output folder: {output_dir}\n")


    # Stage player assets into ad_dir/player/ so they upload with the single s3 cp command
    setup_player_dir(ad_dir)
    player_url = f"{cloudfront_url.rstrip('/')}/{s3_prefix.strip('/')}/audio-descriptions/player"

    html_path = os.path.join(ad_dir, f"{folder_name}.html")

    def write_html(entries):
        """Rewrites the HTML page with whatever entries are complete so far."""
        content = build_ableplayer_html(
            folder_name    = folder_name,
            s3_prefix      = s3_prefix,
            cloudfront_url = cloudfront_url,
            video_entries  = entries,
            player_url     = player_url,
        )
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # Collect metadata for the HTML generator as we process each video
    video_entries = []
    outer_bar = tqdm(mp4_keys, desc="📦 Videos", colour="blue",
                     position=0, leave=True, dynamic_ncols=True,
                     file=sys.stdout)

    for key in outer_bar:
        display_name    = os.path.basename(key).rsplit('.', 1)[0]
        ad_vtt_name     = f"{display_name}_AD.vtt"
        vtt_final       = os.path.join(ad_dir, ad_vtt_name)
        vtt_processing  = os.path.join(ad_dir, f"{display_name}_AD_processing.vtt")
        caption_key     = check_s3_caption_exists(s3, bucket, key)
        outer_bar.set_postfix_str(display_name[:40])

        # ── Resume: VTT already finished from a previous run ──────────────
        if os.path.exists(vtt_final):
            tqdm.write(f"  ⏭️  Skipping (already done): {display_name}")
            video_entries.append({
                "display_name": display_name.replace('_', ' ').title(),
                "video_key":    key,
                "caption_key":  caption_key,
                "ad_vtt_name":  ad_vtt_name,
            })
            write_html(video_entries)
            continue

        # ── Clean up a stale _processing file from an interrupted run ─────
        if os.path.exists(vtt_processing):
            tqdm.write(f"  🧹 Removing stale processing file: {os.path.basename(vtt_processing)}")
            os.unlink(vtt_processing)

        # ── Process: download from S3, analyse, write VTT ─────────────────
        caps     = get_s3_captions(s3, bucket, key)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
                tqdm.write(f"  ⬇️  Downloading: {display_name}")
                s3.download_fileobj(bucket, key, tmp)

            process_video(tmp_path, args, client_params, report_path,
                          output_dir=ad_dir, caps=caps, display_name=display_name)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # ── Add to entry list and immediately update the HTML ──────────────
        video_entries.append({
            "display_name": display_name.replace('_', ' ').title(),
            "video_key":    key,
            "caption_key":  caption_key,
            "ad_vtt_name":  ad_vtt_name,
        })
        write_html(video_entries)
        tqdm.write(f"  🌐 HTML updated ({len(video_entries)}/{len(mp4_keys)} videos)")

    outer_bar.close()  # release tqdm cursor control before prompting
    sys.stdin = sys.__stdin__  # restore original stdin in case it was redirected
    print(f"\n📊 Master Report saved:  {report_path}")
    print(f"🌐 HTML page saved:      {html_path}")


    # ---- Confirm before pushing to S3 / CloudFront ----
    ad_s3_prefix = f"{s3_prefix.strip('/')}/audio-descriptions"
    try:
        answer = input("\nDo you want to update the audio descriptions on S3 and CloudFront? (y/n): ").strip().lower()
    except EOFError:
        answer = 'n'  # non-interactive environment — skip

    if answer != 'y':
        print("  ⏭️  Skipped. Local files saved to:", ad_dir)
        return

    # ---- Upload audio-descriptions/ to S3 ----
    print(f"\n⬆️  Uploading audio-descriptions to s3://{bucket}/{ad_s3_prefix} ...")
    try:
        n = upload_to_s3(s3, ad_dir, bucket, ad_s3_prefix)
        print(f"  ✅ Uploaded {n} file(s)")
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        return

    # ---- Invalidate CloudFront cache ----
    print(f"\n🔄 Invalidating CloudFront cache ...")
    try:
        invalidate_cloudfront(config, s3_prefix)
    except Exception as e:
        print(f"  ❌ CloudFront invalidation failed: {e}")

if __name__ == "__main__": main()