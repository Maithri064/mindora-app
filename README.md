# mindora-app

# 🌿 Mindora — AI Mental Wellness & Emotion Detection Platform

[![Live Demo](https://img.shields.io/badge/Demo-Live_on_Render-2ea44f?style=for-the-badge&logo=render&logoColor=white)](https://mindora-app.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

> **Mindora** is a responsive full-stack mental wellness web application powered by Natural Language Processing (NLP). It allows users to write daily self-reflections, decode emotional patterns in real time, and receive tailored psychological coping exercises and wellness guidance.

🔗 **Live Production URL:** [https://mindora-app.onrender.com](https://mindora-app.onrender.com)

---

## 🌟 Key Features

* **Real-Time Emotion Decoding:** Analyzes user-submitted journal entries and daily reflections to detect underlying emotional states (e.g., Sadness, Joy, Anxiety, Exhaustion) with probability metrics.
* **Personalized Coping Rituals:** Dynamically surfaces evidence-based psychological coping mechanisms, mindfulness prompts, and calming routines tailored to the user's detected emotional state.
* **Daily Wellness Content & Quizzes:** Built-in interactive mental health assessments, mood questionnaires, and curated self-care blogs.
* **Persistent User Storage:** Session tracking, reflection logs, and user activity handled via an integrated SQLite persistence layer.
* **Lightweight Cloud Architecture:** Optimized for serverless and low-memory cloud runtimes (<512 MB footprint) without sacrificing NLP accuracy.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript, Jinja2 Templates |
| **Backend Framework** | Python, Flask, Werkzeug |
| **Database** | SQLite, Flask-SQLAlchemy |
| **NLP & Text Processing** | NLTK (Tokenization, Stopword Filtering, WordNet Lemmatization), NumPy |
| **Production Server** | Gunicorn (WSGI) |
| **Cloud Deployment** | Render Cloud Platform (Automated CI/CD via GitHub) |

---

## 📁 Repository Structure

```text
mindora-app/
├── static/              # CSS stylesheets, UI icons, and frontend assets
├── templates/           # Jinja2 HTML views (landing page, dashboard, quiz)
├── app.py               # Core Flask application, routing, and inference pipeline
├── daily_content.py     # Wellness quotes, psychological rituals, and blogs
├── requirements.txt     # Python runtime dependencies
├── Procfile             # Process declaration for cloud deployment
└── README.md            # Project technical documentation
