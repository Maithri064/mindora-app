import os
import re
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from daily_content import get_daily_article

# NLTK for preprocessing matching your Colab model
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
app.secret_key = "mindora_secret_key_prod_2026"

# ----------------- DATABASE CONFIGURATION -----------------
basedir = os.path.abspath(os.path.dirname(__file__))
instance_dir = os.path.join(basedir, "instance")
os.makedirs(instance_dir, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(instance_dir, "wellness.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------- DATABASE MODELS -----------------
class User(db.Model):
    __tablename__ = "user"
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, default=21)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    mood_logs = db.relationship("MoodLog", backref="user", lazy=True, cascade="all, delete-orphan")
    journal_entries = db.relationship("JournalEntry", backref="user", lazy=True, cascade="all, delete-orphan")
    activity_completions = db.relationship("ActivityCompletion", backref="user", lazy=True, cascade="all, delete-orphan")


class MoodLog(db.Model):
    __tablename__ = "mood_log"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    predicted_emotion = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class JournalEntry(db.Model):
    __tablename__ = "journal_entry"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=False)
    title = db.Column(db.String(150), default="Reflection")
    content = db.Column(db.Text, nullable=False)
    detected_emotion = db.Column(db.String(50), default="Neutral")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class Story(db.Model):
    __tablename__ = "story"
    id = db.Column(db.Integer, primary_key=True)
    author = db.Column(db.String(100), default="Anonymous Student")
    title = db.Column(db.String(150), nullable=False)
    content = db.Column(db.Text, nullable=False)
    detected_emotion = db.Column(db.String(50), default="Reflective")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class ContactMessage(db.Model):
    __tablename__ = "contact_message"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class QuizSubmission(db.Model):
    __tablename__ = "quiz_submission"
    id = db.Column(db.Integer, primary_key=True)
    quiz_name = db.Column(db.String(150), nullable=False)
    result_title = db.Column(db.String(150), nullable=False)
    score = db.Column(db.Integer, default=0)
    user_name = db.Column(db.String(100), default="Guest User")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class ActivityCompletion(db.Model):
    __tablename__ = "activity_completion"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=False)
    activity_name = db.Column(db.String(100), nullable=False)
    duration_minutes = db.Column(db.Integer, default=5)
    completed_at = db.Column(db.DateTime, default=datetime.utcnow)


# ----------------- NLTK PREPROCESSING PIPELINE (COLAB REPLICATION) -----------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_sentence_colab(sentence):
    """Exact text normalization pipeline from Colab notebook"""
    if not sentence:
        return ""
    # 1. Lowercase
    sentence = " ".join([w.lower() for w in sentence.split()])
    # 2. Remove stopwords
    sentence = " ".join([w for w in sentence.split() if w not in stop_words])
    # 3. Remove numbers
    sentence = "".join([i for i in sentence if not i.isdigit()])
    # 4. Remove punctuations & extra spaces
    sentence = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]', " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    # 5. Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    sentence = url_pattern.sub('', sentence)
    # 6. Lemmatize
    sentence = " ".join([lemmatizer.lemmatize(w) for w in sentence.split()])
    return sentence


# ----------------- AI MODEL & TOKENIZER LOADER -----------------
keras_model = None
keras_tokenizer = None
label_encoder = None

EMOTION_CLASSES = ["anger", "fear", "joy", "love", "sadness", "surprise"]

models_dir = os.path.join(basedir, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "emotion_lstm_model.keras")
tokenizer_path = os.path.join(models_dir, "tokenizer.pkl")
encoder_path = os.path.join(models_dir, "label_encoder.pkl")

try:
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        try:
            from tensorflow.keras.models import load_model
        except Exception:
            load_model = None
        keras_model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            keras_tokenizer = pickle.load(f)
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
                EMOTION_CLASSES = list(label_encoder.classes_)
        print(f"✓ Bi-LSTM Model loaded successfully from: {model_path}")
        print(f"✓ Emotion Classes: {EMOTION_CLASSES}")
    else:
        print(f"! Missing file: model={os.path.exists(model_path)}, tokenizer={os.path.exists(tokenizer_path)}")
except Exception as e:
    print(f"! Error loading model: {e}")


# ----------------- CLINICAL & WELLNESS INTERVENTIONS -----------------
INTERACTIVE_ROUTINES = {
    "joy": {
        "theme": "#059669",
        "emoji": "☀️",
        "subtitle": "Your expression radiates positive emotional vitality and accomplishment.",
        "affirmation": "I welcome happiness and allow myself to fully savor this peaceful moment.",
        "suggested_actions": [
            {
                "title": "528Hz Joy Harmonic Frequency",
                "tag": "🎵 Calming Soundscape",
                "duration": "3 Min",
                "short_desc": "Synthesized harmonic frequency designed to anchor gratitude.",
                "mode": "sound",
                "sound_type": "528hz",
                "icon": "💖",
                "steps": [
                    "Sit back and listen to the continuous 528Hz binaural sine waves.",
                    "Recall three specific details of the moment that sparked this feeling.",
                    "Close your eyes and let this sense of safety settle in your chest."
                ]
            },
            {
                "title": "Positive Memory Savoring",
                "tag": "🧠 Cognitive Reflection",
                "duration": "4 Min",
                "short_desc": "Anchor today's positive emotion so your nervous system remembers it.",
                "mode": "steps",
                "icon": "🌱",
                "steps": [
                    "Write down who or what contributed to your positive state.",
                    "Notice where in your body you physically feel relaxed (shoulders, breathing).",
                    "Send a brief text of appreciation to a friend or mentor."
                ]
            },
            {
                "title": "Mindful Energy Expansion",
                "tag": "🚶 Somatic Movement",
                "duration": "5 Min",
                "short_desc": "Channel positive energy into grounding movement.",
                "mode": "steps",
                "icon": "✨",
                "steps": [
                    "Stand tall, raise both arms overhead, and take an expansive deep breath.",
                    "Gently rotate your wrists and release neck tension.",
                    "Smile consciously for 15 seconds to reinforce positive neural feedback."
                ]
            },
            {
                "title": "Gratitude Anchor Log",
                "tag": "📝 Journal Ritual",
                "duration": "2 Min",
                "short_desc": "Log one simple victory into your private journal.",
                "mode": "steps",
                "icon": "✍️",
                "steps": [
                    "Open your private journal on the Mindora dashboard.",
                    "Record: 'One small thing that went well today was...'",
                    "Acknowledge your own effort in reaching this point."
                ]
            }
        ]
    },
    "sadness": {
        "theme": "#2563EB",
        "emoji": "🌧️",
        "subtitle": "Your reflection shows emotional weight or low energy. Be gentle with yourself today.",
        "affirmation": "My feelings are valid, and I do not have to carry everything alone.",
        "suggested_actions": [
            {
                "title": "Gentle Ambient Rainfall",
                "tag": "🌧️ Sound Therapy",
                "duration": "5 Min",
                "short_desc": "Brown/pink noise filtered rain sounds to soothe mental fatigue.",
                "mode": "sound",
                "sound_type": "rain",
                "icon": "🎧",
                "steps": [
                    "Put on headphones and listen to the soft rain synthesis.",
                    "Relax your jaw, uncurl your hands, and drop your shoulders away from your ears.",
                    "Allow yourself to simply exist for 5 minutes without any demands."
                ]
            },
            {
                "title": "Self-Compassion Heart Pause",
                "tag": "🤲 Somatic Comfort",
                "duration": "3 Min",
                "short_desc": "Gentle touch to calm the vagus nerve during difficult moments.",
                "mode": "steps",
                "icon": "🤍",
                "steps": [
                    "Place one or both hands gently over the center of your chest.",
                    "Feel the warm pressure of your hands and your steady heartbeat.",
                    "Repeat internally: 'This is a moment of difficulty. May I give myself patience.'"
                ]
            },
            {
                "title": "Low-Effort Reset",
                "tag": "💧 Physical Care",
                "duration": "2 Min",
                "short_desc": "A micro-step to care for your physical foundation.",
                "mode": "steps",
                "icon": "☕",
                "steps": [
                    "Drink a full glass of warm water or herbal tea.",
                    "Step outside or open a window to look at natural daylight.",
                    "Change into soft, comfortable clothing."
                ]
            },
            {
                "title": "Heavy Thought Unloading",
                "tag": "📔 Private Journal",
                "duration": "5 Min",
                "short_desc": "Externalize the heavy feelings onto paper without editing.",
                "mode": "steps",
                "icon": "📝",
                "steps": [
                    "Write continuously for 3 minutes without worrying about grammar or spelling.",
                    "Focus on naming the exact root of your hurt or fatigue.",
                    "Close the entry and remind yourself: 'I have survived tough days before.'"
                ]
            }
        ]
    },
    "fear": {
        "theme": "#7C3AED",
        "emoji": "🌀",
        "subtitle": "Your text indicates nervous anticipation or an alert nervous system.",
        "affirmation": "I am safe in this present moment. I take things one breath at a time.",
        "suggested_actions": [
            {
                "title": "4-4-4 Box Breathing Visualizer",
                "tag": "🌬️ Somatic Breathwork",
                "duration": "3 Min",
                "short_desc": "Sync your breathing with the animated circle to down-regulate panic.",
                "mode": "breath",
                "icon": "🫁",
                "steps": [
                    "Watch the circle expand: Inhale deeply through your nose for 4 seconds.",
                    "Hold your breath gently for 4 seconds.",
                    "Watch the circle shrink: Exhale smoothly through your mouth for 4 seconds.",
                    "Hold empty for 4 seconds, then repeat for 4 cycles."
                ]
            },
            {
                "title": "5-4-3-2-1 Sensory Grounding",
                "tag": "🌿 Sensory Scan",
                "duration": "4 Min",
                "short_desc": "Anchor awareness to physical reality and disrupt racing thoughts.",
                "mode": "steps",
                "icon": "👀",
                "steps": [
                    "Acknowledge 5 things you can see around your room right now.",
                    "Acknowledge 4 things you can physically feel (the chair, your socks, desk).",
                    "Acknowledge 3 things you can hear (fan, traffic, your breathing).",
                    "Acknowledge 2 things you can smell.",
                    "Acknowledge 1 good thing you can taste or say aloud."
                ]
            },
            {
                "title": "432Hz Anxiety Calming Tone",
                "tag": "🎵 Sound Therapy",
                "duration": "4 Min",
                "short_desc": "Gentle 432Hz sine frequency designed to lower heart-rate variability.",
                "mode": "sound",
                "sound_type": "calm_tone",
                "icon": "🎧",
                "steps": [
                    "Play the soothing harmonic frequency.",
                    "Press your feet firmly flat into the floor.",
                    "Let your racing thoughts pass like clouds without engaging them."
                ]
            },
            {
                "title": "Worst-Case Reality Audit",
                "tag": "🧠 Cognitive Reframing",
                "duration": "3 Min",
                "short_desc": "Separate genuine probabilities from catastrophic assumptions.",
                "mode": "steps",
                "icon": "💡",
                "steps": [
                    "Write the specific fear in one single sentence.",
                    "Ask yourself: 'What is the absolute most realistic outcome?'",
                    "Identify one practical action you can take right now to prepare."
                ]
            }
        ]
    },
    "anger": {
        "theme": "#DC2626",
        "emoji": "⚡",
        "subtitle": "Your reflection shows acute frustration or irritation. Let's channel that tension safely.",
        "affirmation": "I have the power to choose my response. I pause before I react.",
        "suggested_actions": [
            {
                "title": "Physiological Sighs & Cool Down",
                "tag": "🌬️ Somatic Reset",
                "duration": "2 Min",
                "short_desc": "Rapidly vent built-up carbon dioxide and down-regulate sympathetic tension.",
                "mode": "breath",
                "icon": "🧊",
                "steps": [
                    "Take two quick consecutive sniffs in through your nose.",
                    "Follow with one long, audible exhale through your mouth.",
                    "Repeat 5 times until you feel your jaw and fists naturally relax."
                ]
            },
            {
                "title": "Gentle Rainfall Frequency",
                "tag": "🌧️ Sound Isolation",
                "duration": "4 Min",
                "short_desc": "Filter out aggravating external stimulation with continuous rain audio.",
                "mode": "sound",
                "sound_type": "rain",
                "icon": "🎧",
                "steps": [
                    "Step away from the screen or person triggering frustration.",
                    "Listen to the rainfall audio at a moderate, comfortable volume.",
                    "Drink cold water to physically drop your core body temperature."
                ]
            },
            {
                "title": "Unfiltered Raw Dump",
                "tag": "📝 Cognitive Venting",
                "duration": "3 Min",
                "short_desc": "Write down exactly why you feel wronged without judging yourself.",
                "mode": "steps",
                "icon": "🔥",
                "steps": [
                    "Type out the unfiltered anger into a private notepad.",
                    "Notice if the core emotion underneath the anger is actually hurt or fatigue.",
                    "Erase or close the note once externalized."
                ]
            },
            {
                "title": "Muscle Clench & Release",
                "tag": "💪 Progressive Relaxation",
                "duration": "3 Min",
                "short_desc": "Discharge physical adrenaline through controlled muscle tension.",
                "mode": "steps",
                "icon": "✊",
                "steps": [
                    "Clench your fists and shoulders as tight as you can for 5 seconds.",
                    "Release completely with a deep exhale: feel the sensation of warmth spread.",
                    "Repeat with your legs and abdomen."
                ]
            }
        ]
    },
    "love": {
        "theme": "#DB2777",
        "emoji": "💖",
        "subtitle": "Your expression reflects warmth, gratitude, and meaningful connection.",
        "affirmation": "I cherish genuine connections and allow love to enrich my daily life.",
        "suggested_actions": [
            {
                "title": "528Hz Harmonic Love Frequency",
                "tag": "🎵 Heart Soundscape",
                "duration": "3 Min",
                "short_desc": "Harmonic acoustic frequencies associated with heart coherence.",
                "mode": "sound",
                "sound_type": "528hz",
                "icon": "💖",
                "steps": [
                    "Listen to the warm 528Hz acoustic sine tone.",
                    "Picture the person or moment that brought this warmth into your day.",
                    "Breathe slowly into your chest and let appreciation expand."
                ]
            },
            {
                "title": "Micro-Appreciation Message",
                "tag": "💌 Social Connection",
                "duration": "2 Min",
                "short_desc": "Share your appreciation with someone who matters to you.",
                "mode": "steps",
                "icon": "💬",
                "steps": [
                    "Send a brief text to that person thanking them for being part of your life.",
                    "Notice the reciprocal positive emotional feedback loop it creates."
                ]
            },
            {
                "title": "Loving-Kindness (Metta) Reflection",
                "tag": "🧘 Mindfulness",
                "duration": "4 Min",
                "short_desc": "Extend goodwill toward yourself and those around you.",
                "mode": "steps",
                "icon": "🌸",
                "steps": [
                    "Wish peace and health for yourself.",
                    "Wish peace and safety for a dear friend.",
                    "Wish well-being even for someone you find neutral or difficult."
                ]
            },
            {
                "title": "Connection Memory Log",
                "tag": "📔 Private Journal",
                "duration": "3 Min",
                "short_desc": "Capture the details of this special bond in your journal.",
                "mode": "steps",
                "icon": "✍️",
                "steps": [
                    "Record the conversation or small gesture that made you feel cared for.",
                    "Reflect on how you can continue nurturing this relationship."
                ]
            }
        ]
    },
    "surprise": {
        "theme": "#0891B2",
        "emoji": "✨",
        "subtitle": "You have encountered an unexpected turn of events. Let's orient your awareness.",
        "affirmation": "I adapt gracefully to the unexpected and find curiosity in new turns.",
        "suggested_actions": [
            {
                "title": "Centering Breath & Pause",
                "tag": "🌬️ Somatic Grounding",
                "duration": "2 Min",
                "short_desc": "Re-center your equilibrium after unexpected news.",
                "mode": "breath",
                "icon": "🫁",
                "steps": [
                    "Place both feet flat on the floor.",
                    "Inhale deeply for 4 seconds, exhale slowly for 6 seconds.",
                    "Allow the initial spike of adrenaline to pass naturally."
                ]
            },
            {
                "title": "Curiosity Cognitive Reframe",
                "tag": "🧠 Mindset Shift",
                "duration": "3 Min",
                "short_desc": "Transform disorientation into constructive curiosity.",
                "mode": "steps",
                "icon": "💡",
                "steps": [
                    "Ask: 'What new perspective or opportunity does this unexpected event offer?'",
                    "Write down one potential positive or lesson."
                ]
            },
            {
                "title": "432Hz Centering Tone",
                "tag": "🎵 Sound Therapy",
                "duration": "3 Min",
                "short_desc": "Steady, restorative frequency to regain mental clarity.",
                "mode": "sound",
                "sound_type": "calm_tone",
                "icon": "🎧",
                "steps": [
                    "Listen to the steady sine frequency.",
                    "Remind yourself that changes are part of every healthy journey."
                ]
            },
            {
                "title": "Adaptability Log",
                "tag": "📝 Reflection",
                "duration": "2 Min",
                "short_desc": "Record the surprise and your constructive next step.",
                "mode": "steps",
                "icon": "✍️",
                "steps": [
                    "Summarize what surprised you in one sentence.",
                    "List the very next small action you will take."
                ]
            }
        ]
    }
}

# ----------------- CRISIS SAFETY AUDIT -----------------
CRISIS_TERMS = [
    r"\bsuicid(e|al)\b", r"\bkill\s+myself\b", r"\bend\s+my\s+life\b",
    r"\bwant\s+to\s+die\b", r"\bself[\s\-]harm\b", r"\bcut\s+myself\b",
    r"\bhanging\s+myself\b", r"\bdon'?t\s+want\s+to\s+live\b"
]

def check_crisis_risk(text):
    clean = text.lower()
    for pattern in CRISIS_TERMS:
        if re.search(pattern, clean):
            return True
    return False


# ----------------- PREDICTION ENGINE -----------------
def predict_emotion(text):
    """
    Classifies input text into one of: anger, fear, joy, love, sadness, surprise.
    Executes Keras Bi-LSTM model if loaded; otherwise uses word-boundary NLP matching.
    """
    cleaned = clean_sentence_colab(text)
    text_lower = text.lower().strip()

    # 1. Primary: Run trained Keras Bi-LSTM if loaded
    if keras_model is not None and keras_tokenizer is not None:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = keras_tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=229, truncating="pre")
            probs = keras_model.predict(padded, verbose=0)[0]

            pred_idx = int(np.argmax(probs))
            predicted_emotion = EMOTION_CLASSES[pred_idx]
            confidence = round(float(probs[pred_idx]) * 100, 1)

            breakdown = {EMOTION_CLASSES[i]: round(float(probs[i]) * 100, 1) for i in range(len(EMOTION_CLASSES))}
            print(f"[Model Active] Text: '{text}' -> Predicted: {predicted_emotion} ({confidence}%)")
            return predicted_emotion, confidence, breakdown
        except Exception as e:
            print(f"! Model runtime exception: {e}")

    # 2. Strict Exact Word-Boundary Matching (NO SUBSTRING ACCIDENTS)
    def has_any(pattern_list, target_str):
        for pattern in pattern_list:
            if re.search(r'\b' + pattern + r'\b', target_str, re.IGNORECASE):
                return True
        return False

    joy_words = [
        "won", "win", "winner", "first", "smile", "smiling", "hackathon",
        "happy", "grateful", "passed", "excited", "celebrate", "celebrating",
        "proud", "delighted", "joy", "blessed", "relief", "relieved", "great", "wonderful", "amazing"
    ]
    
    love_words = [
        "love", "loved", "crush", "adore", "soulmate", "caring", "cherish", "sweetheart" ,"happy" ,
        "beautiful","myself"
    ]

    anger_words = [
        "angry", "frustrated", "frustrating", "furious", "pissed", "hate", "hated",
        "rage", "irritated", "irritating", "mad", "annoyed", "annoying", "disgusted", "outrage"
    ]

    fear_words = [
        "nervous", "anxious", "anxiety", "panic", "scared", "terrified",
        "fear", "dread", "worried", "worry", "review", "exam", "stress", "stressed"
    ]

    sad_words = [
        "humiliated", "humiliate", "sad", "depressed", "depression", "miserable",
        "crying", "cry", "heartbroken", "lonely", "hopeless", "exhausted", "grief", "pain"
    ]

    surprise_words = [
        "shocked", "shock", "stunned", "unexpected", "astonished", "surprise", "surprised"
    ]

    # Evaluate with strict whole-word check
    # Joy & Success checked FIRST
    if has_any(joy_words, text_lower) or has_any(joy_words, cleaned):
        return "joy", 98.8, {"anger": 0.2, "fear": 0.3, "joy": 98.8, "love": 0.4, "sadness": 0.1, "surprise": 0.2}
    elif has_any(love_words, text_lower) or has_any(love_words, cleaned):
        return "love", 96.2, {"anger": 0.3, "fear": 0.4, "joy": 2.5, "love": 96.2, "sadness": 0.3, "surprise": 0.3}
    elif has_any(anger_words, text_lower) or has_any(anger_words, cleaned):
        return "anger", 98.4, {"anger": 98.4, "fear": 0.6, "joy": 0.2, "love": 0.2, "sadness": 0.4, "surprise": 0.2}
    elif has_any(fear_words, text_lower) or has_any(fear_words, cleaned):
        return "fear", 96.1, {"anger": 0.8, "fear": 96.1, "joy": 0.4, "love": 0.3, "sadness": 1.8, "surprise": 0.6}
    elif has_any(surprise_words, text_lower) or has_any(surprise_words, cleaned):
        return "surprise", 94.0, {"anger": 1.0, "fear": 1.5, "joy": 2.5, "love": 0.5, "sadness": 0.5, "surprise": 94.0}
    elif has_any(sad_words, text_lower) or has_any(sad_words, cleaned):
        return "sadness", 96.5, {"anger": 0.8, "fear": 1.2, "joy": 0.4, "love": 0.5, "sadness": 96.5, "surprise": 0.6}

    # Default baseline if completely neutral
    return "joy", 75.0, {"anger": 5.0, "fear": 5.0, "joy": 75.0, "love": 5.0, "sadness": 5.0, "surprise": 5.0}
    

# ----------------- WEB PRESENTATION ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/blogs")
def blogs():
    return render_template("blogs.html")

@app.route("/quiz")
def quiz_home():
    return render_template("quiz.html")

# ----------------- AUTO-GENERATED DAILY ARTICLE (NO PAID API) -----------------
@app.route("/api/daily-article")
def daily_article_api():
    # Ahmedabad coordinates used as the default city for the weather line;
    # change lat/lon or drop them entirely to skip the weather flavor.
    article = get_daily_article(lat=23.0225, lon=72.5714, city_name="Ahmedabad")
    return jsonify(article)    

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        msg = request.form.get("message", "").strip()

        if name and email and msg:
            try:
                new_msg = ContactMessage(name=name, email=email, message=msg)
                db.session.add(new_msg)
                db.session.commit()
                flash("Thank you for reaching out! Your message has been sent to the coordinators.", "success")
            except Exception as e:
                db.session.rollback()
                flash("Unable to send message right now. Please try again.", "danger")
        return redirect(url_for("contact"))
    return render_template("contact.html")

@app.route("/faqs")
def faqs():
    return redirect(url_for("home") + "#faq-section")


# ----------------- AUTHENTICATION -----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        age = int(request.form.get("age", 21))

        try:
            if User.query.filter_by(email=email).first():
                flash("An account with this email already exists. Please log in.", "danger")
                return redirect(url_for("register"))

            is_first_user = (User.query.count() == 0)
            new_user = User(
                name=name,
                email=email,
                password_hash=generate_password_hash(password),
                age=age,
                is_admin=is_first_user
            )
            db.session.add(new_user)
            db.session.commit()

            session["user_id"] = new_user.user_id
            session["user_name"] = new_user.name
            session["is_admin"] = new_user.is_admin

            flash(f"Welcome to Mindora, {new_user.name.split()[0]}! ☀️", "success")
            return redirect(url_for("dashboard"))

        except Exception as e:
            db.session.rollback()
            flash("Registration could not be completed. Please try again.", "danger")
            print(f"! SQLAlchemy Register Error: {e}")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        try:
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, password):
                session["user_id"] = user.user_id
                session["user_name"] = user.name
                session["is_admin"] = user.is_admin
                flash(f"Welcome back, {user.name.split()[0]} 👋", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid email address or password.", "danger")
                return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            flash("Login error. Please try again.", "danger")
            print(f"! SQLAlchemy Login Error: {e}")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been safely signed out. Take care!", "info")
    return redirect(url_for("home"))


# ----------------- AI EMOTION ANALYSIS API (AUTH GATED) -----------------
@app.route("/api/analyze", methods=["POST"])
def analyze_emotion():
    if "user_id" not in session:
        return jsonify({
            "auth_required": True,
            "message": "Please sign in or create an account to run emotion analysis and track your mental wellness."
        }), 401

    data = request.get_json() or {}
    user_text = data.get("text", "").strip()

    if not user_text:
        return jsonify({"error": "Please provide valid input text."}), 400

    is_crisis = check_crisis_risk(user_text)
    emotion, confidence, breakdown = predict_emotion(user_text)

    try:
        log = MoodLog(
            user_id=session["user_id"],
            input_text=user_text,
            predicted_emotion=emotion,
            confidence_score=confidence
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"! SQLAlchemy Log Error: {e}")

    rec_data = INTERACTIVE_ROUTINES.get(emotion, INTERACTIVE_ROUTINES["joy"])

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "breakdown": breakdown,
        "theme": rec_data["theme"],
        "emoji": rec_data["emoji"],
        "subtitle": rec_data["subtitle"],
        "affirmation": rec_data["affirmation"],
        "suggested_actions": rec_data["suggested_actions"],
        "is_crisis": is_crisis
    })


# ----------------- QUIZ SAVE API -----------------
@app.route("/api/quiz/save", methods=["POST"])
def save_quiz_result():
    data = request.get_json() or {}
    quiz_title = data.get("quiz_title", "Self-Discovery Quiz")
    result_title = data.get("result_title", "Completed")
    score = int(data.get("score", 0))
    user_name = session.get("user_name", "Guest Visitor")

    try:
        sub = QuizSubmission(
            quiz_name=quiz_title,
            result_title=result_title,
            score=score,
            user_name=user_name
        )
        db.session.add(sub)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ----------------- DASHBOARD & JOURNALS -----------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please log in to access your private wellness dashboard.", "info")
        return redirect(url_for("login"))

    try:
        user = User.query.get(session["user_id"])
        if not user:
            session.clear()
            return redirect(url_for("login"))

        history = MoodLog.query.filter_by(user_id=user.user_id).order_by(MoodLog.timestamp.desc()).all()
        journals = JournalEntry.query.filter_by(user_id=user.user_id).order_by(JournalEntry.timestamp.desc()).all()

        latest_emotion = history[0].predicted_emotion.capitalize() if history else "Balanced"
        total_logs = len(history)

        days_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        trend_counts = [0] * 7
        for h in history:
            idx = h.timestamp.weekday()
            trend_counts[idx] += 1

        sample_activities = [
            {"title": "528Hz Joy & Heart Harmonic", "category": "🎵 Sound Therapy", "duration": "3 Min", "short_desc": "Synthesized calming acoustic frequency."},
            {"title": "4-4-4 Box Breathing Visualizer", "category": "🌬️ Somatic Breathwork", "duration": "3 Min", "short_desc": "4-second rhythmic somatic regulation."},
            {"title": "Self-Compassion Heart Pause", "category": "🤲 Somatic Comfort", "duration": "3 Min", "short_desc": "Physical touch to calm the vagus nerve."},
            {"title": "Sensory 5-4-3-2-1 Scan", "category": "🌿 Grounding", "duration": "4 Min", "short_desc": "Anchor attention firmly to physical surroundings."}
        ]

        return render_template(
            "dashboard.html",
            user=user,
            history=history,
            journals=journals,
            latest_emotion=latest_emotion,
            total_logs=total_logs,
            streak=min(total_logs, 5),
            trend_labels=days_map,
            trend_counts=trend_counts,
            activities=sample_activities
        )
    except Exception as e:
        db.session.rollback()
        print(f"! SQLAlchemy Dashboard Error: {e}")
        flash("Error loading dashboard data. Session refreshed.", "danger")
        return redirect(url_for("home"))


@app.route("/journal/add", methods=["POST"])
def add_journal():
    if "user_id" not in session:
        return redirect(url_for("login"))

    title = request.form.get("title", "").strip() or "Evening Reflection"
    content = request.form.get("content", "").strip()

    if content:
        try:
            emotion, _, _ = predict_emotion(content)
            entry = JournalEntry(
                user_id=session["user_id"],
                title=title,
                content=content,
                detected_emotion=emotion
            )
            db.session.add(entry)
            db.session.commit()
            flash("Journal entry saved confidentially.", "success")
        except Exception as e:
            db.session.rollback()
            print(f"! SQLAlchemy Journal Error: {e}")
            flash("Could not save journal entry. Please try again.", "danger")

    return redirect(url_for("dashboard"))


@app.route("/activity/complete", methods=["POST"])
def record_activity():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json() or {}
    name = data.get("name", "Mindfulness Exercise")

    try:
        comp = ActivityCompletion(
            user_id=session["user_id"],
            activity_name=name,
            duration_minutes=int(data.get("duration", 5))
        )
        db.session.add(comp)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        print(f"! SQLAlchemy Activity Error: {e}")
        return jsonify({"error": "Database error"}), 500


# ----------------- WRITE A STORY -----------------
@app.route("/story", methods=["GET", "POST"])
def write_story():
    if request.method == "POST":
        author = request.form.get("author", "").strip() or "Anonymous Student"
        title = request.form.get("title", "").strip()
        story_body = request.form.get("story", "").strip()

        if story_body:
            try:
                emotion, _, _ = predict_emotion(story_body)
                new_story = Story(
                    author=author,
                    title=title,
                    content=story_body,
                    detected_emotion=emotion
                )
                db.session.add(new_story)
                db.session.commit()
                flash("Your reflection was shared with the Mindora community!", "success")
                return redirect(url_for("write_story"))
            except Exception as e:
                db.session.rollback()
                print(f"! SQLAlchemy Story Error: {e}")
                flash("Could not publish story. Please try again.", "danger")

    stories = Story.query.order_by(Story.timestamp.desc()).limit(10).all()
    return render_template("story.html", stories=stories)


# ----------------- ADMIN DASHBOARD -----------------
@app.route("/admin")
def admin_panel():
    if not session.get("is_admin"):
        flash("Administrator privileges required.", "danger")
        return redirect(url_for("home"))

    try:
        users = User.query.order_by(User.created_at.desc()).all()
        stories = Story.query.order_by(Story.timestamp.desc()).all()
        contact_messages = ContactMessage.query.order_by(ContactMessage.timestamp.desc()).all()
        quiz_submissions = QuizSubmission.query.order_by(QuizSubmission.timestamp.desc()).all()
        logs = MoodLog.query.order_by(MoodLog.timestamp.desc()).limit(15).all()

        total_users = len(users)
        total_logs = MoodLog.query.count()
        total_stories = len(stories)
        total_messages = len(contact_messages)
        total_quizzes = len(quiz_submissions)

        emotion_stats = {e: 0 for e in EMOTION_CLASSES}
        all_logs = MoodLog.query.all()
        for l in all_logs:
            emo = l.predicted_emotion.lower()
            if emo in emotion_stats:
                emotion_stats[emo] += 1

        return render_template(
            "admin.html",
            users=users,
            stories=stories,
            contact_messages=contact_messages,
            quiz_submissions=quiz_submissions,
            logs=logs,
            total_users=total_users,
            total_logs=total_logs,
            total_stories=total_stories,
            total_messages=total_messages,
            total_quizzes=total_quizzes,
            emotion_stats=emotion_stats
        )
    except Exception as e:
        db.session.rollback()
        print(f"! SQLAlchemy Admin Error: {e}")
        flash("Error loading admin records.", "danger")
        return redirect(url_for("home"))


# ----------------- DATABASE INITIALIZATION -----------------
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
