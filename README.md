# MediReportAI

A Dockerized Flask application that leverages Azure OpenAI services to analyze medical lab test reports via Retrieval-Augmented Generation (RAG). Users can securely upload PDF lab reports, ask natural-language questions about their results, and receive comprehensive, formatted responses.

---

## Table of Contents

- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Deployment](#deployment)  
- [Environment Variables](#environment-variables)  
- [Security](#security)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## Features

- Secure user authentication (Flask-Login + SQLite).  
- Upload multiple PDF lab reports.  
- Document chunking & embedding via Azure OpenAI Embeddings.  
- Similarity search (cosine similarity) over embedded report excerpts.  
- GPT-based RAG response generation for medical insights.  
- Markdown-to-HTML conversion for rich, styled results (Tailwind Typography).  
- Fully containerized with Docker & Docker Compose.  

---

## Tech Stack

- **Backend**: Python 3.13, Flask, Flask-Login  
- **AI Services**: Azure OpenAI (Embeddings & Chat)  
- **Vector Storage**: In-memory Python list (for demo)  
- **Database**: SQLite (for user credentials)  
- **Frontend**: Jinja2 templates, Tailwind CSS, Markdown  
- **Containerization**: Docker, Docker Compose  
- **Reverse Proxy & SSL**: Caddy  

---

## Prerequisites

- Docker & Docker Compose installed on your machine.  
- An Azure subscription with an Azure OpenAI resource.  
- Public or whitelisted network access for your Azure OpenAI endpoint.  
- License or free-tier access within $30/month budget on Azure.  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/medireportai.git
   cd medireportai
   ```

2. **Copy example environment file**  
   ```bash
   cp .env.example .env
   # Edit `.env` with your actual credentials
   ```

3. **Build & start containers**  
   ```bash
   docker-compose up --build -d
   ```

4. **Access the app**  
   Open your browser at `https://ai.yourdomain.com` (or `http://localhost:5000` if testing locally).

---

## Configuration

Edit the `.env` file to set:

```dotenv
FLASK_SECRET_KEY=your_random_secret_key
ADMIN_USER=admin
ADMIN_PASS=securepassword
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2023-05-15
EMBEDDING_DEPLOYMENT=embeddings
CHAT_DEPLOYMENT=chat
```

> **Note**: `.env` contains sensitive credentials and **must NOT** be committed to version control. A `.env.example` file is provided for reference.

---

## Usage

1. **Login** with your admin credentials.  
2. **Upload** one or more PDF lab reports.  
3. **Enter** a question (e.g., “Explain my HBs Ag result”).  
4. **Review** the formatted analysis displayed on screen.  

---

## Project Structure

```
medireportai/
├── app/
│   ├── main.py           # Flask application entrypoint
│   ├── templates/        # Jinja2 HTML templates
│   └── static/           # Static assets (CSS, images)
├── instance/
│   └── users.db          # SQLite database (user credentials)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Deployment

- Use Docker Compose to orchestrate the Flask app and Caddy reverse proxy.  
- Caddy handles SSL (internal or Let’s Encrypt) and routes `ai.yourdomain.com` to the Flask container.  
- Ensure the `caddy-network` is created as an external network for both stacks.

---

## Environment Variables

See [Configuration](#configuration).  
Sensitive values (keys, passwords) reside in `.env`, which is excluded from git.

---

## Security

- **Authentication** protects Azure API credits.  
- **Firewall** or **VNet** rules restrict Azure OpenAI access to your server’s IP.

---

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit your changes (`git commit -am 'Add my feature'`)  
4. Push to branch (`git push origin feature/my-feature`)  
5. Open a Pull Request  

---

## License

MIT License. See [LICENSE](LICENSE) for details.
