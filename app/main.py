import os
import io
import math
import sqlite3
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import openai
import markdown

# Load environment variables
load_dotenv()

# Flask app (instance folder one level up)
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Inject current year into templates
@app.context_processor
def inject_year():
    return {'year': datetime.now().year}

# Flask-Login setup
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# Path to SQLite DB (mounted at /app/instance)
DATABASE = os.path.abspath(os.path.join(app.root_path, os.pardir, "instance", "users.db"))

def init_db():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    # Create users table
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )"""
    )
    # Insert default admin user if not exists
    admin_user = os.getenv("ADMIN_USER")
    admin_pass = os.getenv("ADMIN_PASS")
    c.execute("SELECT * FROM users WHERE username=?", (admin_user,))
    if not c.fetchone():
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (admin_user, admin_pass)
        )
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id_, username, password):
        self.id = id_
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, username, password FROM users WHERE id=?", (int(user_id),))
    row = c.fetchone()
    conn.close()
    if row:
        return User(*row)
    return None

# Login route
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute(
            "SELECT id, username, password FROM users WHERE username=? AND password=?",
            (username, password)
        )
        row = c.fetchone()
        conn.close()
        if row:
            user = User(*row)
            login_user(user)
            return redirect(url_for("chat"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")

# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# Azure OpenAI setup using official openai package
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")

# In-memory vector store
vector_store = []

def chunk_text(text, max_len=1000):
    """Split text into chunks of approx max_len characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        if end < len(text):
            split = text.rfind("\n", start, end)
            if split == -1:
                split = text.rfind(" ", start, end)
            if split != -1:
                end = split
        chunks.append(text[start:end])
        start = end
    return chunks


def get_embedding(text):
    resp = openai.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    # resp.data is a list of embedding objects, each has an `.embedding` attribute
    return resp.data[0].embedding


def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(y*y for y in b))
    return dot / (mag_a * mag_b)

# Main chat route
@app.route("/", methods=["GET", "POST"])
@login_required
def chat():
    answer = None
    if request.method == "POST":
        # PDF upload and embedding
        files = request.files.getlist("pdfs")
        vector_store.clear()
        for file in files:
            if file and file.filename.lower().endswith('.pdf'):
                text = ''
                reader = PdfReader(io.BytesIO(file.read()))
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                chunks = chunk_text(text)
                for chunk in chunks:
                    emb = get_embedding(chunk)
                    vector_store.append({"chunk": chunk, "embedding": emb})
        # Query and RAG
        query = request.form.get("query")
        if query and vector_store:
            q_emb = get_embedding(query)
            sims = [
                {"chunk": item["chunk"], "score": cosine_similarity(q_emb, item["embedding"]) }
                for item in vector_store
            ]
            sims = sorted(sims, key=lambda x: x['score'], reverse=True)[:3]
            context = "\n\n".join(item['chunk'] for item in sims)
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in analyzing medical lab test reports."},
                {"role": "user", "content": f"Here are the relevant excerpts:\n{context}\n\nQuestion: {query}"}
            ]
            chat_resp = openai.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=messages,
                max_tokens=500
            )

            raw_md = chat_resp.choices[0].message.content
            answer_html = markdown.markdown(raw_md, extensions=["fenced_code", "tables"])
    return render_template("chat.html", answer_html=answer_html)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
