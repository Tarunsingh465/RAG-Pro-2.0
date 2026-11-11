"""
Zero-API Conversational Intent Engine.
Intercepts all generic small-talk, status inquiries, greetings, bot-identity,
capabilities, courtesy, and help questions in-memory in < 0.0005 seconds (0.5ms)
with ZERO external API calls, ZERO network latency, and ZERO credit consumption.
"""

import re
from typing import Optional


def normalize_query(query: str) -> str:
    """
    Strip punctuation and extra whitespace, converting to lower case.
    Apostrophes are removed directly so 'how's' -> 'hows', 'what's' -> 'whats'.
    """
    cleaned = re.sub(r"['’`]", "", query)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


# Intent patterns mapped to deterministic, professional Markdown responses
INTENT_PATTERNS = [
    # 1. Compound Greeting & Status ("hey how are you", "hello how are you doing", "hi how's it going")
    {
        "id": "compound_greeting_status",
        "patterns": [
            r"^(hey|hi|hello|greetings)\s+(there\s+)?(how\s+are\s+you|how\s+r\s+u|how\s+are\s+things|how\s+is\s+it\s+going|hows\s+it\s+going|how\s+do\s+you\s+do|how\s+are\s+you\s+doing|whats\s+up|what\s+is\s+up)(\s+(today|there|doing))?$",
            r"\b(hey|hi|hello)\s+(how\s+are\s+you|how\s+r\s+u|how\s+are\s+things|how\s+is\s+it\s+going|hows\s+it\s+going|how\s+are\s+you\s+doing)\b",
        ],
        "response": (
            "Hey there! I'm doing great and fully operational. 😊\n\n"
            "I'm ready to help you analyze your documents, evaluate resumes, "
            "or solve engineering problems across CS, Mechanical, Civil, Chemical, and Electrical domains. "
            "What would you like to explore today?"
        ),
    },

    # 2. Pure Status Inquiries ("how are you", "how are you doing", "how's it going", "how are things", "what's up")
    {
        "id": "status_inquiry",
        "patterns": [
            r"^(how\s+are\s+you|how\s+r\s+u|how\s+do\s+you\s+do|how\s+are\s+things|how\s+is\s+it\s+going|hows\s+it\s+going|how\s+are\s+you\s+doing|whats\s+up|what\s+is\s+up|sup)(\s+(today|doing|there|doing\s+today))?$",
            r"\b(how\s+are\s+you|how\s+are\s+you\s+doing|how\s+is\s+it\s+going|hows\s+it\s+going)\b",
        ],
        "response": (
            "I'm doing well, thank you for asking! 🚀\n\n"
            "I am ready to help you with your documents and technical tasks. "
            "Feel free to upload a document on the left sidebar, or ask me an engineering question directly!"
        ),
    },

    # 3. Purpose & Identity ("what is your purpose", "who are you", "what is your role", "tell me about yourself")
    {
        "id": "purpose_and_identity",
        "patterns": [
            r"^(what\s+is\s+your\s+purpose|whats\s+your\s+purpose|what\s+is\s+your\s+role|whats\s+your\s+role|who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|whats\s+your\s+name|what\s+are\s+you\s+called|tell\s+me\s+about\s+yourself|who\s+created\s+you|who\s+made\s+you|what\s+is\s+this\s+app|what\s+is\s+this\s+project|what\s+is\s+rag\s*pro|what\s+does\s+rag\s*pro\s+do|why\s+were\s+you\s+created)$",
            r"\b(what\s+is\s+your\s+purpose|whats\s+your\s+purpose|what\s+is\s+your\s+role|whats\s+your\s+role|who\s+are\s+you|tell\s+me\s+about\s+yourself|what\s+is\s+your\s+name|whats\s+your\s+name)\b",
        ],
        "response": (
            "I am **RAG-Pro 2.0**, an AI-powered Enterprise Knowledge & Industrial Intelligence Assistant.\n\n"
            "### 🎯 My Purpose:\n"
            "- 📄 **Document Intelligence:** Ingest, parse, and search technical PDFs, Word docs, CSVs, and notes.\n"
            "- ⚙️ **Engineering Problem-Solving:** Provide solutions, governing equations ($...$), and boundary conditions across CS, Mechanical, Civil, Chemical, and Electrical engineering.\n"
            "- 🔍 **Industrial Troubleshooting:** Diagnose faults and equipment failures using Root Cause Analysis (RCA) and compliance standards (ASME, IEEE, IS, ISO).\n"
            "- 👤 **Candidate Evaluation:** Perform structured resume assessments and hiring feasibility breakdowns.\n\n"
            "You can upload your documents in the sidebar or ask technical questions right here!"
        ),
    },

    # 4. Capabilities & Features ("what can you do", "what do you do", "what are your features", "what can i ask")
    {
        "id": "capabilities",
        "patterns": [
            r"^(what\s+can\s+you\s+do|what\s+do\s+you\s+do|what\s+can\s+i\s+ask(\s+you)?|what\s+are\s+your\s+capabilities|what\s+are\s+your\s+features|how\s+can\s+you\s+help(\s+me)?|how\s+can\s+you\s+assist(\s+me)?|what\s+documents\s+can\s+you\s+read|what\s+file\s+types\s+do\s+you\s+support)$",
            r"\b(what\s+can\s+you\s+do|what\s+are\s+your\s+capabilities|what\s+are\s+your\s+features)\b",
        ],
        "response": (
            "Here is what I can do for you:\n\n"
            "1. **Document Q&A:** Upload any technical document, report, or resume, and ask questions. I retrieve exact excerpts with page citations.\n"
            "2. **Engineering Calculations:** Ask questions involving formulas, derivations, design standards, and technical specifications.\n"
            "3. **Zero-Cost Conversations:** Common conversational questions and greetings respond instantly without wasting any API credits.\n"
            "4. **Multi-Format Ingestion:** Supports PDF, DOCX, PPTX, TXT, MD, CSV, and XLSX.\n\n"
            "Try uploading a document on the left, or ask me an engineering query!"
        ),
    },

    # 5. Help & Quick Start ("help", "help me", "how to use", "instructions", "guide me")
    {
        "id": "help_and_guide",
        "patterns": [
            r"^(help|help\s+me|how\s+to\s+use|how\s+do\s+i\s+use|how\s+do\s+i\s+use\s+this|instructions|guide\s+me|where\s+do\s+i\s+start|what\s+can\s+i\s+do\s+here)$",
        ],
        "response": (
            "### 🛠️ Quick Start Guide:\n\n"
            "1. **Upload Documents:** Use the left sidebar to select your files (PDF, DOCX, CSV, etc.) and click **Upload & Index**.\n"
            "2. **Ask Questions:** Type your question in the chat bar below. I will search through your documents and provide answers with page citations.\n"
            "3. **Engineering Queries:** Even without documents uploaded, feel free to ask technical concepts, formulas, or troubleshooting steps!"
        ),
    },

    # 6. Courtesy & Acknowledgments ("thanks", "thank you", "cool", "awesome", "great", "ok", "got it")
    {
        "id": "courtesy_thanks",
        "patterns": [
            r"\b(thanks|thank\s+you|thanku|thx|appreciate\s+it)\b",
            r"^(ok|okay|cool|awesome|great|nice|perfect|got\s+it|sounds\s+good|alright|fine)$",
        ],
        "response": (
            "You're very welcome! 😊 Feel free to ask any other questions or upload more documents whenever you're ready."
        ),
    },

    # 7. Casual Humor ("tell me a joke", "say something funny", "make me laugh")
    {
        "id": "casual_humor",
        "patterns": [
            r"\b(tell\s+me\s+a\s+joke|make\s+me\s+laugh|say\s+something\s+funny|tell\s+a\s+joke)\b",
        ],
        "response": (
            "Why do programmers prefer dark mode? 💻\n\n"
            "Because light attracts bugs! 🐛😄\n\n"
            "How can I assist you with your documents or engineering problems today?"
        ),
    },

    # 8. AI / Bot Identity ("are you an ai", "are you a bot", "are you human", "what model are you")
    {
        "id": "ai_bot_identity",
        "patterns": [
            r"\bare\s+you\s+(an\s+ai|a\s+bot|a\s+robot|human|real)\b",
            r"\b(what\s+model\s+are\s+you|what\s+llm\s+are\s+you|what\s+llm)\b",
        ],
        "response": (
            "Yes, I am an AI assistant built on an advanced Retrieval-Augmented Generation (RAG) architecture. "
            "I combine semantic vector search with large language model reasoning to deliver accurate, grounded technical answers."
        ),
    },

    # 9. Farewells ("bye", "goodbye", "see you", "take care", "good night")
    {
        "id": "farewell",
        "patterns": [
            r"\b(bye|goodbye|see\s+you|cya|adios|take\s+care|good\s+night|have\s+a\s+nice\s+day)\b",
        ],
        "response": (
            "Goodbye! Have a great day, and feel free to return whenever you need assistance analyzing documents or solving engineering problems! 👋"
        ),
    },

    # 10. Simple Direct Greetings ("hi", "hello", "hey", "good morning")
    {
        "id": "simple_greeting",
        "patterns": [
            r"^(hi|hello|hey|hola|howdy|greetings|good\s+morning|good\s+afternoon|good\s+evening)$",
            r"^(hi|hello|hey)\s+(there|buddy|assistant|friend)$",
        ],
        "response": (
            "Hello! I am your RAG & Industrial Intelligence Assistant. "
            "How can I help you analyze your documents or solve a technical problem today?"
        ),
    },
]


def match_conversational_intent(query: str) -> Optional[dict]:
    """
    Check if a query matches any zero-API conversational intent.
    Returns dict with 'id' and 'response' if matched, or None.
    Execution time: < 0.0005 seconds (0.5 ms).
    """
    clean = normalize_query(query)

    if not clean:
        return {
            "id": "empty_query",
            "response": "Hello! How can I assist you today? Please ask a question or upload a document to get started.",
        }

    for entry in INTENT_PATTERNS:
        for pat in entry["patterns"]:
            if re.search(pat, clean):
                return {
                    "id": entry["id"],
                    "response": entry["response"],
                }

    return None
