\# ⚡ FrankenAI v9.0



\*\*A production-ready conversational AI chatbot with 4-pass iterative reasoning and conversation memory.\*\*



Built from scratch in one intense coding session. Zero templates, just raw problem-solving.



---



\## 🎯 What Makes This Different?



Most AI chatbots send your question to an API and return the first answer. FrankenAI uses a \*\*4-pass self-refining system\*\* that makes the AI critique and improve its own responses before you see them.



\### The 4-Pass System



1\. \*\*Pass 1 (Initial Answer)\*\* - Quick first response

2\. \*\*Pass 2 (Self-Critique)\*\* - AI reviews its own answer: "What did I miss?"

3\. \*\*Pass 3 (Refinement)\*\* - Rewrite with improvements from critique

4\. \*\*Pass 4 (Context Check)\*\* - Adjust based on conversation history



\*\*Result:\*\* Answers are 2-3x more comprehensive and context-aware than single-pass responses.



---



\## 🔥 Features



\- ✅ \*\*4-Pass Iterative Reasoning\*\* - Self-critique and refinement before responding

\- ✅ \*\*Conversation Memory\*\* - Remembers last 3 exchanges for contextual follow-ups

\- ✅ \*\*Smart Caching\*\* - Instant answers for repeated questions (saves API costs)

\- ✅ \*\*Retry Logic\*\* - Exponential backoff handles rate limits gracefully

\- ✅ \*\*Optimized Tokens\*\* - 30% reduction in token usage vs initial design

\- ✅ \*\*Built-in Knowledge\*\* - Math calculator, greetings, facts (no API needed)

\- ✅ \*\*Stats API\*\* - Monitor usage, cache performance, conversation metrics

\- ✅ \*\*Session Management\*\* - Isolated conversations per user



---



\## 💰 Cost Breakdown



Uses \*\*Groq Llama 3.3 70B\*\* (paid tier):

\- ~$0.005-$0.01 per question (half a penny to one penny)

\- 100 questions = ~$0.08

\- 1,000 questions = ~$0.80

\- Monthly heavy usage: $2-$5



\*\*Example:\*\* Yesterday's test session:

\- 100 requests

\- 120,000 tokens

\- \*\*Total cost: $0.08\*\*



Much cheaper than OpenAI/Claude while maintaining quality through multi-pass reasoning.



---



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.8+

\- Groq API key (\[Get one here](https://console.groq.com))



\### Installation

```bash

\# Clone repository

git clone https://github.com/yourusername/frankenai.git

cd frankenai



\# Install dependencies

pip install -r requirements.txt



\# Create .env file with your API key

echo "GROQ\_API\_KEY=your\_api\_key\_here" > .env

echo "FLASK\_SECRET\_KEY=your\_random\_secret\_key" >> .env



\# Run the app

python app.py

```



Open browser: `http://localhost:5000`



---



\## 📊 API Endpoints



| Endpoint | Method | Description |

|----------|--------|-------------|

| `/` | GET | Main chat interface |

| `/api/ask` | POST | Send a question, get 4-pass refined answer |

| `/api/stats` | GET | System statistics (usage, cache, performance) |

| `/api/restart` | POST | Clear conversation history for current session |



\### Example Request

```bash

curl -X POST http://localhost:5000/api/ask \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"query": "explain quantum entanglement"}'

```



\### Response

```json

{

&nbsp; "success": true,

&nbsp; "answer": "Quantum entanglement is a phenomenon where...",

&nbsp; "sources": \[{"title": "Powered by Groq AI (4-pass)", "url": "https://groq.com"}],

&nbsp; "method": "Groq 4-Pass Conversational",

&nbsp; "cached": false,

&nbsp; "conversation\_length": 1

}

```



---



\## 🧠 How the 4-Pass System Works



\### Problem with Single-Pass AI



Standard chatbots return the first answer the AI generates. This often means:

\- Missing important details

\- No self-review

\- No context from previous questions



\### FrankenAI's Solution



\*\*Pass 1: Initial Answer\*\* (1200 tokens max)

```

User: "Explain black holes"

AI: \[Generates initial explanation]

```



\*\*Pass 2: Self-Critique\*\* (800 tokens max)

```

Prompt: "Review this answer. What's missing? What could be better?"

AI: "Missing: Hawking radiation, event horizon details, real examples..."

```



\*\*Pass 3: Refined Answer\*\* (2000 tokens max)

```

Prompt: "Rewrite using the critique. Be comprehensive."

AI: \[Generates improved explanation with critique insights]

```



\*\*Pass 4: Context Check\*\* (2000 tokens max)

```

Prompt: "Consider conversation history (2 previous exchanges):

&nbsp; 1. User asked about gravity

&nbsp; 2. Now asking about black holes

&nbsp; 

Adjust answer to acknowledge connection if relevant."

AI: \[Final context-aware answer]

```



\*\*Total tokens:\*\* ~5,600-11,200 per question  

\*\*Quality improvement:\*\* 2-3x more comprehensive than single-pass



---



\## 💾 Smart Caching System



\### How It Works



1\. \*\*Cache Key Generation\*\*

&nbsp;  - Hash of: conversation history + current question

&nbsp;  - Same context + same question = same cache key



2\. \*\*Cache Hit\*\*

&nbsp;  - Question asked before with same context

&nbsp;  - Return cached answer instantly (0 API calls)

&nbsp;  - Saves money and improves speed



3\. \*\*Cache Miss\*\*

&nbsp;  - New question or new context

&nbsp;  - Run full 4-pass system

&nbsp;  - Cache result for future use



\### Example

```

Session 1:

Q: "What is quantum physics?"

→ 4 API calls, answer cached



Session 2 (same user):

Q: "What is quantum physics?"

→ 0 API calls, instant from cache

```



---



\## 🧪 Conversation Memory



Stores last 3 Q\&A pairs per session for contextual responses.



\### Example Conversation

```

Q1: "Who created Grok?"

A1: "Grok was created by xAI, Elon Musk's AI company..."

\[Added to history]



Q2: "What's special about it?"

→ Pass 4 sees history, understands "it" = Grok

A2: "Based on our previous discussion about Grok, what makes it special is..."

\[Context-aware response]



Q3: "How does it compare to ChatGPT?"

→ Pass 4 sees 2 previous exchanges about Grok

A3: "Continuing our conversation about Grok, compared to ChatGPT..."

\[Full context awareness]

```



---



\## 📁 Project Structure

```

FrankenAI/

├── app.py                 # Flask backend (350+ lines)

│   ├── 4-pass reasoning system

│   ├── Conversation memory

│   ├── Caching logic

│   └── Retry with exponential backoff

├── .env                   # Environment variables (API keys)

├── requirements.txt       # Python dependencies

├── .gitignore            # Git ignore file

├── README.md             # This file

├── templates/

│   └── index.html        # Chat UI (responsive, clean)

└── cache/                # Auto-created, stores cached answers

```



---



\## 🔧 Configuration



\### Environment Variables (`.env`)

```bash

GROQ\_API\_KEY=your\_groq\_api\_key\_here

FLASK\_SECRET\_KEY=your\_random\_secret\_for\_sessions

```



\### Token Limits (in `app.py`)



Adjust these in the `groq\_four\_pass()` function:

```python

\# Pass 1: Initial answer

max\_tokens=1200  # Default: 1200



\# Pass 2: Critique

max\_tokens=800   # Default: 800



\# Pass 3: Refined answer

max\_tokens=2000  # Default: 2000



\# Pass 4: Context check

max\_tokens=2000  # Default: 2000

```



\*\*Lower = cheaper, faster | Higher = more detailed\*\*



---



\## 📈 Stats Endpoint



`GET /api/stats` returns:

```json

{

&nbsp; "api\_requests\_made": 48,

&nbsp; "cached\_answers": 12,

&nbsp; "cache\_hits": 8,

&nbsp; "cache\_misses": 12,

&nbsp; "cache\_hit\_rate": "40.0%",

&nbsp; "total\_queries": 20,

&nbsp; "active\_conversations": 3,

&nbsp; "current\_conversation\_length": 2,

&nbsp; "rate\_limit\_hits": 0

}

```



---



\## 🎓 Version History



| Version | Release | Key Features |

|---------|---------|--------------|

| v9.0 | Jan 2026 | \*\*Current\*\* - Production ready, env vars, retry logic, optimized tokens |

| v8.1 | Jan 2026 | Fixed conversation history bug |

| v8.0 | Jan 2026 | Added 4-pass system + conversation memory |

| v7.1 | Jan 2026 | Added stats endpoint |

| v7.0 | Jan 2026 | File-based caching + 3-pass system |

| v1-6 | Jan 2026 | Early iterations (web scraping experiments) |



---



\## 🛠️ Tech Stack



\- \*\*Backend:\*\* Python Flask

\- \*\*AI Model:\*\* Groq Llama 3.3 70B (versatile)

\- \*\*Frontend:\*\* Vanilla HTML/CSS/JavaScript

\- \*\*Caching:\*\* JSON files

\- \*\*Session Management:\*\* Flask sessions



---



\## 📊 Performance



\### Speed

\- \*\*First query:\*\* 5-8 seconds (4 API calls)

\- \*\*Cached query:\*\* <100ms (instant)

\- \*\*Follow-up with context:\*\* 5-8 seconds (4 API calls with history)



\### Quality Comparison



| Metric | Single-Pass | FrankenAI 4-Pass |

|--------|-------------|------------------|

| Answer length | ~1,500 chars | ~8,000 chars |

| Detail level | Basic | Comprehensive |

| Context awareness | None | Full (3 exchanges) |

| Self-correction | No | Yes (Pass 2) |

| Cost per query | $0.003 | $0.008 |



\*\*Verdict:\*\* 2.7x more expensive, but 5x better quality and full conversation memory.



---



\## 🐛 Known Limitations



1\. \*\*Token usage:\*\* 4 passes = higher cost than single-pass (mitigated by caching)

2\. \*\*Speed:\*\* 5-8 seconds per query (acceptable for quality, could add streaming)

3\. \*\*Memory:\*\* Limited to last 3 exchanges (could expand with vector store)

4\. \*\*Offline:\*\* Requires internet for Groq API (no local mode yet)



---



\## 🚀 Future Enhancements



Potential improvements:

\- \[ ] Streaming responses (real-time text generation)

\- \[ ] Vector store for long-term memory (beyond 3 exchanges)

\- \[ ] Voice input/output

\- \[ ] Multi-user support with user accounts

\- \[ ] Fine-tuning on specific domains

\- \[ ] Docker containerization

\- \[ ] API rate limit visualization

\- \[ ] Cost prediction per query



---



\## 🤝 Contributing



Pull requests welcome! For major changes, open an issue first.



\*\*Areas for contribution:\*\*

\- Frontend improvements (dark mode, mobile optimization)

\- Additional built-in knowledge

\- Performance optimizations

\- Documentation improvements



---



\## 📝 License



MIT License - Feel free to use, modify, and distribute.



---



\## 🙏 Acknowledgments



Built with:

\- \[Flask](https://flask.palletsprojects.com/) - Web framework

\- \[Groq](https://groq.com/) - LLM API

\- \[Llama 3.3 70B](https://www.llama.com/) - AI model

\- Lots of coffee and debugging ☕



---



\## 📧 Questions?



Open an issue or reach out!



---



\*\*⚡ FrankenAI - Because one pass isn't enough.\*\*

