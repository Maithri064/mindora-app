"""
daily_content.py
-----------------
Generates a "fresh" daily wellness article for Mindora WITHOUT calling any
paid external AI API.

How it stays "fresh" every day, forever, for free:
  1. A hand-written bank of wellness articles (same format/voice as the
     existing blogs.html scenario cards).
  2. Two days of the week (Monday / Sunday) always show a locked, highly
     relevant article ("Monday Dread", "Sunday Scaries").
  3. The other five days rotate deterministically through the remaining
     bank using the day-of-year, so the whole town sees the same article
     on the same day, and it naturally cycles back around later in the year.
  4. An OPTIONAL, completely free weather lookup (Open-Meteo — no API key,
     no cost, no signup) is used to add a live, locally-relevant opening
     line ("It's a rainy Wednesday in Ahmedabad..."). If the request fails
     or is skipped, the article still renders fine with a generic line.

Nothing here depends on OpenAI/Anthropic/any paid text-generation API.
"""

import urllib.request
import json
from datetime import datetime

# ----------------------------------------------------------------------
# 1. THE CONTENT BANK
# ----------------------------------------------------------------------
DAILY_ARTICLES = {
    "monday_dread": {
        "tag": "Daily Insight · Monday",
        "title": "The Monday Scaries: Why Your Body Dreads the Week Before Your Mind Does",
        "content": """
            <p><strong>The Real-Life Scene:</strong> It's Monday morning. Before you've even opened your eyes,
            there's already a low hum of dread in your stomach — the week stretching out ahead like a to-do list
            with no end. You're not sad, exactly. Just braced.</p>
            <h4 style="margin-top:14px;color:#332014;">Why Your Body Does This:</h4>
            <p>Researchers call this the "Sunday Night Spillover" — your nervous system starts anticipating
            Monday's demands up to 12 hours early, releasing small amounts of cortisol before you're even awake.
            It's not weakness; it's your body trying to prepare you.</p>
            <h4 style="margin-top:14px;color:#332014;">The Reset Protocol:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Pick One Small Win:</strong> Before checking messages, decide on ONE small task
                you'll finish first. Momentum beats motivation.</li>
                <li><strong>Delay the Inbox:</strong> Give yourself 20 minutes of slow morning before opening
                email or chat apps — the dread often comes from anticipated demands, not real ones yet.</li>
                <li><strong>Name It Out Loud:</strong> Simply saying "this is Monday dread, it always fades by
                10am" reduces its intensity — labeling emotions calms the amygdala.</li>
            </ul>
        """
    },
    "sunday_scaries": {
        "tag": "Daily Insight · Sunday",
        "title": "The Sunday Scaries: Grieving a Weekend That Isn't Even Over Yet",
        "content": """
            <p><strong>The Real-Life Scene:</strong> It's only 4 PM on Sunday, technically hours of freedom left,
            but a heavy feeling has already crept in — half dread, half grief for a weekend that "went too fast."</p>
            <h4 style="margin-top:14px;color:#332014;">What's Actually Happening:</h4>
            <p>This is anticipatory anxiety, not present-moment sadness. Your mind has already time-traveled to
            Monday's demands, robbing you of the hours you still have.</p>
            <h4 style="margin-top:14px;color:#332014;">The Reclaiming Ritual:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Do One "Sunday-Only" Thing:</strong> A slow meal, a show, a walk — something that
                marks the day as still belonging to you, not to Monday.</li>
                <li><strong>Prep Just Enough:</strong> Lay out tomorrow's essentials for 5 minutes, then
                consciously stop — over-preparing feeds the anxiety loop instead of calming it.</li>
                <li><strong>Reframe the Ending:</strong> Replace "the weekend is over" with "I get to start
                fresh tomorrow" — same fact, kinder frame.</li>
            </ul>
        """
    },
    "midweek_slump": {
        "tag": "Daily Insight · Midweek",
        "title": "The Wednesday Wall: Why Motivation Crashes Right in the Middle",
        "content": """
            <p><strong>The Real-Life Scene:</strong> You're not tired enough to justify a break, but not fresh
            enough to focus either. Everything feels slightly heavier than it should by Wednesday afternoon.</p>
            <h4 style="margin-top:14px;color:#332014;">Why This Happens:</h4>
            <p>Cognitive load accumulates across the week. By the midpoint, your prefrontal cortex — responsible
            for focus and decision-making — is running on a deficit, even if you slept fine.</p>
            <h4 style="margin-top:14px;color:#332014;">The Midweek Reset:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Micro-Break Rule:</strong> Step away from your screen for 5 full minutes every 90
                minutes — true rest, not scrolling.</li>
                <li><strong>Lower the Bar Temporarily:</strong> Wednesday doesn't need your best output — just
                forward motion. Permission to coast a little is not failure.</li>
                <li><strong>Hydrate Before You Caffeinate:</strong> Mild dehydration mimics mental fatigue almost
                exactly — a full glass of water often helps more than another coffee.</li>
            </ul>
        """
    },
    "screen_fatigue": {
        "tag": "Daily Insight · Digital Life",
        "title": "Zoom Face: The Exhaustion of Back-to-Back Video Calls",
        "content": """
            <p><strong>The Real-Life Scene:</strong> After four video calls in a row, you feel strangely more
            drained than a full day of in-person conversation would leave you — even though you barely moved.</p>
            <h4 style="margin-top:14px;color:#332014;">The Hidden Cost:</h4>
            <p>Video calls force constant, unnatural eye contact and force your brain to process delayed audio,
            frozen frames, and your own mirrored face simultaneously — a cognitive load in-person talk doesn't have.</p>
            <h4 style="margin-top:14px;color:#332014;">The Recovery Steps:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Hide Self-View:</strong> Most platforms let you hide your own video tile — watching
                yourself constantly adds real measurable fatigue.</li>
                <li><strong>Audio-Only When You Can:</strong> If the meeting allows it, turn your camera off for
                one call today and just listen.</li>
                <li><strong>20-20-20 Eye Reset:</strong> Every 20 minutes, look at something 20 feet away for
                20 seconds to release visual strain.</li>
            </ul>
        """
    },
    "rainy_day_mood": {
        "tag": "Daily Insight · Weather & Mood",
        "title": "Grey Sky, Grey Mood: Why Rainy Days Hit Different",
        "content": """
            <p><strong>The Real-Life Scene:</strong> Nothing bad happened today, but the overcast sky outside
            seems to have seeped into your mood — a little flatter, a little slower, a little harder to start.</p>
            <h4 style="margin-top:14px;color:#332014;">The Biology of It:</h4>
            <p>Lower light levels reduce serotonin production and disrupt your circadian rhythm slightly, even on
            a single cloudy day — your brain is quite literally responding to less sunlight.</p>
            <h4 style="margin-top:14px;color:#332014;">The Brighten-Up Routine:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Maximize Indoor Light:</strong> Sit near the brightest window you have, or turn on
                a warm lamp near your workspace.</li>
                <li><strong>Move Your Body Briefly:</strong> Even 5 minutes of movement counteracts the
                low-light energy dip more effectively than caffeine.</li>
                <li><strong>Lower the Bar, Gently:</strong> It's okay for a rainy day to be a slower day — match
                your expectations to your actual energy, not yesterday's sunny one.</li>
            </ul>
        """
    },
    "decision_fatigue": {
        "tag": "Daily Insight · Mental Load",
        "title": "\"I Don't Even Know What I Want for Dinner\": Understanding Decision Fatigue",
        "content": """
            <p><strong>The Real-Life Scene:</strong> By evening, even small choices — what to eat, what to
            watch, whether to reply to that message — feel unreasonably difficult. You just want someone else
            to decide for you.</p>
            <h4 style="margin-top:14px;color:#332014;">Why This Happens:</h4>
            <p>Every decision, big or small, draws from the same limited mental resource. By the end of a full
            day of choices, that resource is genuinely depleted — this is measurable, not imagined.</p>
            <h4 style="margin-top:14px;color:#332014;">The Low-Effort Fix:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Pre-Decide When Fresh:</strong> Choose tomorrow's outfit or meal in the morning,
                when your decision reserves are full.</li>
                <li><strong>Reduce Options on Purpose:</strong> Pick from 2 choices instead of endless ones —
                fewer options genuinely feels like relief, not restriction.</li>
                <li><strong>It's Okay to Default:</strong> Reusing yesterday's answer ("same dinner as
                Tuesday") is a valid, healthy shortcut — not a failure of adulting.</li>
            </ul>
        """
    },
    "unanswered_message": {
        "tag": "Daily Insight · Digital Anxiety",
        "title": "The Unread Notification Badge That Won't Leave You Alone",
        "content": """
            <p><strong>The Real-Life Scene:</strong> A single unanswered email or message sits in your notifications
            all day. You've seen it four times. Each glance adds a tiny spike of unease, even though you know
            exactly what it says.</p>
            <h4 style="margin-top:14px;color:#332014;">The Loop Behind It:</h4>
            <p>Unfinished tasks create what psychologists call the Zeigarnik Effect — your brain keeps
            "pinging" you about open loops far more than completed ones, regardless of how small the task is.</p>
            <h4 style="margin-top:14px;color:#332014;">Closing the Loop:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>2-Minute Rule:</strong> If replying takes under 2 minutes, do it the moment you see
                it — closing loops immediately prevents the accumulation.</li>
                <li><strong>Schedule It Instead of Ignoring It:</strong> If you can't reply now, write yourself
                "Reply at 5 PM" — a plan quiets the open loop almost as well as finishing it.</li>
                <li><strong>Mute, Don't Delete:</strong> Silence the badge notification until your scheduled
                time — visibility alone re-triggers the anxiety loop.</li>
            </ul>
        """
    },
    "comparison_at_work": {
        "tag": "Daily Insight · Career & Study",
        "title": "Everyone Else Seems to Have It Figured Out (They Don't)",
        "content": """
            <p><strong>The Real-Life Scene:</strong> A classmate posts about an internship offer, or a coworker
            casually mentions finishing something you've been stuck on for days. A quiet "what's wrong with me?"
            creeps in.</p>
            <h4 style="margin-top:14px;color:#332014;">The Comparison Trap:</h4>
            <p>You're comparing your full, messy behind-the-scenes process to someone else's single polished
            highlight — an inherently unfair comparison your brain makes anyway.</p>
            <h4 style="margin-top:14px;color:#332014;">The Grounding Reframe:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Name the Distortion:</strong> Simply noting "this is a highlight-reel comparison"
                weakens its emotional grip.</li>
                <li><strong>Check Your Own Timeline:</strong> List one thing you understand today that you
                didn't a month ago — progress is rarely visible in the moment.</li>
                <li><strong>Limit the Exposure:</strong> A short break from the feed that triggered the
                comparison is a valid form of self-care, not avoidance.</li>
            </ul>
        """
    },
    "overstimulated_weekend": {
        "tag": "Daily Insight · Social Battery",
        "title": "Too Much Weekend: Why Fun Plans Can Still Leave You Drained",
        "content": """
            <p><strong>The Real-Life Scene:</strong> You had a genuinely good weekend — plans, people, maybe even
            things you enjoyed. And yet by Sunday evening you feel wrung out rather than recharged.</p>
            <h4 style="margin-top:14px;color:#332014;">Why Good Things Can Still Drain You:</h4>
            <p>Enjoyable stimulation still uses the same social and sensory "battery" as stressful stimulation.
            Back-to-back plans, even fun ones, don't leave room for the nervous system to fully reset.</p>
            <h4 style="margin-top:14px;color:#332014;">The Recharge Plan:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Protect One Unscheduled Block:</strong> Even 30 minutes with absolutely no plan
                helps your nervous system downshift.</li>
                <li><strong>Lower Sensory Input:</strong> Dim lights, quiet music, or silence for a stretch —
                give your senses a genuine rest, not just your schedule.</li>
                <li><strong>It's Not Ingratitude:</strong> Needing quiet after good things doesn't mean you
                didn't enjoy them — it means you're human, not antisocial.</li>
            </ul>
        """
    },
    "tired_despite_sleep": {
        "tag": "Daily Insight · Energy & Rest",
        "title": "Slept 8 Hours, Still Exhausted: The Difference Between Rest and Recovery",
        "content": """
            <p><strong>The Real-Life Scene:</strong> You technically got a full night's sleep, but you wake up
            feeling like you barely closed your eyes — heavy limbs, foggy thinking, no real sense of being
            restored.</p>
            <h4 style="margin-top:14px;color:#332014;">Sleep Quantity Isn't Everything:</h4>
            <p>Mental and emotional fatigue from unresolved stress doesn't fully clear during sleep the way
            physical tiredness does — your body rested, but your nervous system may not have.</p>
            <h4 style="margin-top:14px;color:#332014;">The Recovery Add-Ons:</h4>
            <ul style="padding-left:20px;margin-top:6px;">
                <li><strong>Morning Sunlight:</strong> 5–10 minutes of natural light shortly after waking helps
                reset your circadian rhythm more than extra sleep alone.</li>
                <li><strong>Name the Open Stress:</strong> Write down one unresolved worry — unspoken stress
                often shows up as physical fatigue the next day.</li>
                <li><strong>Gentle Movement Over Caffeine:</strong> A short walk or stretch increases real
                alertness more sustainably than a second cup of coffee.</li>
            </ul>
        """
    },
}

# Days that always show a fixed, contextually locked article
# Python's weekday(): Monday=0 ... Sunday=6
WEEKDAY_LOCKED = {
    0: "monday_dread",
    6: "sunday_scaries",
}

# The rotating pool = everything except the two locked-day articles
_ROTATING_KEYS = [k for k in DAILY_ARTICLES.keys() if k not in WEEKDAY_LOCKED.values()]


# ----------------------------------------------------------------------
# 2. OPTIONAL FREE WEATHER FLAVOR (Open-Meteo — no key, no cost)
# ----------------------------------------------------------------------
_WEATHER_MOOD_MAP = {
    range(0, 4):   "clear skies",
    range(45, 49): "a misty haze",
    range(51, 68): "a light drizzle",
    range(71, 78): "gentle snowfall",
    range(80, 83): "passing showers",
    range(95, 100): "a thunderstorm rumbling nearby",
}


def _describe_weather_code(code):
    for rng, label in _WEATHER_MOOD_MAP.items():
        if code in rng:
            return label
    return "mixed conditions overhead"


def get_weather_line(lat=23.0225, lon=72.5714, city_name="your city"):
    """
    Free, no-key weather lookup via Open-Meteo. Returns a short, human
    opening line, or None if the request fails for any reason (no
    internet in this sandbox, timeout, etc.) so the article still
    renders fine without it.
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        code = data.get("current_weather", {}).get("weathercode")
        temp = data.get("current_weather", {}).get("temperature")
        if code is None:
            return None
        description = _describe_weather_code(code)
        temp_txt = f" around {round(temp)}°C" if temp is not None else ""
        return f"It's {description}{temp_txt} in {city_name} right now — worth noticing how that's shaping your mood today."
    except Exception:
        return None


# ----------------------------------------------------------------------
# 3. PUBLIC FUNCTION
# ----------------------------------------------------------------------
def get_daily_article(lat=None, lon=None, city_name="your city"):
    """
    Returns a dict: { tag, title, content, date_label, weather_line }
    Deterministic per calendar day — everyone sees the same article on
    the same date, and it automatically rotates without any manual work
    or paid API calls.
    """
    today = datetime.now()
    weekday = today.weekday()
    day_of_year = today.timetuple().tm_yday

    if weekday in WEEKDAY_LOCKED:
        key = WEEKDAY_LOCKED[weekday]
    else:
        key = _ROTATING_KEYS[day_of_year % len(_ROTATING_KEYS)]

    article = dict(DAILY_ARTICLES[key])
    article["date_label"] = today.strftime("%A, %B %d, %Y")

    weather_line = None
    if lat is not None and lon is not None:
        weather_line = get_weather_line(lat, lon, city_name)
    article["weather_line"] = weather_line

    return article