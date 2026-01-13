from flask import Flask, request, jsonify, render_template, session
import re
import json
import hashlib
import time
import random
from datetime import datetime
from groq import Groq
import os
from collections import deque
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'frankenai_secret_key_2026')

# Initialize Groq client from environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Create a .env file with your API key.")

groq_client = Groq(api_key=GROQ_API_KEY)

# Cache directory
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Request counter
request_counter = {
    'count': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'rate_limit_hits': 0
}

# Conversation history storage (per session)
conversation_history = {}

# KNOWLEDGE BASE
KNOWLEDGE_BASE = {
    'greetings': {
        'patterns': [r'\bhi\b', r'\bhello\b', r'\bhey\b', r'\bsup\b', r'\byo\b'],
        'response': "Hey! I'm FrankenAI v9.0 - Production-ready 4-pass conversational AI with retry logic and optimized token usage!"
    },
    'earth': {
        'patterns': [r'earth circumference', r'earth around'],
        'response': "Earth circumference: 24,901 miles (40,075 km) at equator"
    },
    'thanks': {
        'patterns': [r'\bthank', r'\bthanks\b', r'\bthx\b'],
        'response': "You're welcome!"
    }
}

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    return session['session_id']

def get_conversation_history(session_id):
    """Get conversation history for session"""
    if session_id not in conversation_history:
        conversation_history[session_id] = deque(maxlen=3)
    return conversation_history[session_id]

def add_to_history(session_id, query, answer):
    """Add Q&A to conversation history"""
    history = get_conversation_history(session_id)
    history.append({'query': query, 'answer': answer[:500]})

def get_cache_key(query, session_id):
    """Generate cache key including session context"""
    history = get_conversation_history(session_id)
    context_str = json.dumps([h['query'] for h in history])
    return hashlib.md5(f"{context_str}_{query}".encode()).hexdigest()

def get_cached_answer(query, session_id):
    """Check if answer exists in cache"""
    try:
        cache_key = get_cache_key(query, session_id)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                request_counter['cache_hits'] += 1
                print(f"  -> Cache HIT! Saved 4 API calls")
                return cached
        
        request_counter['cache_misses'] += 1
    except Exception as e:
        print(f"Cache read error: {e}")
    return None

def save_to_cache(query, session_id, result):
    """Save answer to cache"""
    try:
        cache_key = get_cache_key(query, session_id)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  -> Cached answer for future use")
    except Exception as e:
        print(f"Cache write error: {e}")

def check_knowledge_base(query):
    """Check built-in knowledge"""
    lower = query.lower()
    for category, data in KNOWLEDGE_BASE.items():
        if 'patterns' in data:
            for pattern in data['patterns']:
                if re.search(pattern, lower):
                    return {'answer': data['response'], 'sources': [], 'method': 'Built-in', 'cached': False}
    
    # Math calculator
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)', query)
    if math_match:
        a, op, b = float(math_match.group(1)), math_match.group(2), float(math_match.group(3))
        if op == '+': result = a + b
        elif op == '-': result = a - b
        elif op == '*': result = a * b
        elif op == '/': result = a / b if b != 0 else 'undefined'
        return {'answer': f"{a} {op} {b} = {result}", 'sources': [], 'method': 'Calculator', 'cached': False}
    
    return None

def rate_limited_api_call(func, *args, **kwargs):
    """
    Wrapper for Groq API calls with exponential backoff retry logic
    Handles rate limits gracefully
    """
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if 'rate limit' in error_str or 'tokens per minute' in error_str or '429' in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  -> Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    request_counter['rate_limit_hits'] += 1
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Rate limit retries exhausted after {max_retries} attempts")
            else:
                # Not a rate limit error, raise immediately
                raise e
    
    raise Exception("API call failed after all retries")

def groq_four_pass(query, session_id):
    """
    4-pass reasoning with conversational context
    Optimized token usage with reduced max_tokens
    """
    try:
        history = get_conversation_history(session_id)
        history_count = len(history)
        
        # Build context string from history
        context_str = ""
        if history:
            context_str = f"\n\nConversation history ({history_count} previous exchanges):\n"
            for i, item in enumerate(history, 1):
                context_str += f"{i}. User: {item['query']}\n   AI: {item['answer'][:150]}...\n"
        
        # PASS 1: Initial answer (optimized tokens)
        print(f"  -> Pass 1/4: Initial answer (history: {history_count} items)...")
        pass1_response = rate_limited_api_call(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are a helpful AI assistant. Provide detailed, accurate answers.{context_str}"},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        initial_answer = pass1_response.choices[0].message.content
        request_counter['count'] += 1
        print(f"  -> Pass 1 complete ({len(initial_answer)} chars)")
        
        # PASS 2: Self-critique (optimized tokens)
        print(f"  -> Pass 2/4: Self-critique...")
        pass2_response = rate_limited_api_call(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a critical reviewer. Identify what's missing or could be improved. Be concise."},
                {"role": "user", "content": f"Question: {query}\n\nAnswer:\n{initial_answer}\n\nProvide specific improvements."}
            ],
            temperature=0.5,
            max_tokens=800
        )
        critique = pass2_response.choices[0].message.content
        request_counter['count'] += 1
        print(f"  -> Pass 2 complete ({len(critique)} chars)")
        
        # PASS 3: Refined answer
        print(f"  -> Pass 3/4: Refined answer...")
        pass3_response = rate_limited_api_call(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Provide the most comprehensive answer using the critique."},
                {"role": "user", "content": f"Question: {query}\n\nFirst answer:\n{initial_answer}\n\nCritique:\n{critique}\n\nProvide the BEST answer."}
            ],
            temperature=0.6,
            max_tokens=2000
        )
        refined_answer = pass3_response.choices[0].message.content
        request_counter['count'] += 1
        print(f"  -> Pass 3 complete ({len(refined_answer)} chars)")
        
        # PASS 4: Context check and adjustment
        print(f"  -> Pass 4/4: Context awareness check...")
        
        if history_count > 0:
            context_prompt = f"""Current question: {query}

Current answer:
{refined_answer}

Conversation history ({history_count} previous exchanges):
{context_str}

IMPORTANT: You have {history_count} previous exchanges. Analyze if this question relates to previous context.

If related: Acknowledge connection and reference previous context.
If standalone: Keep answer as is.

Provide final, context-aware answer."""
        else:
            context_prompt = f"""Current question: {query}

Current answer:
{refined_answer}

FIRST question (history: 0). Provide answer as written."""
        
        pass4_response = rate_limited_api_call(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Conversational AI. Consider history count."},
                {"role": "user", "content": context_prompt}
            ],
            temperature=0.6,
            max_tokens=2000
        )
        final_answer = pass4_response.choices[0].message.content
        request_counter['count'] += 1
        print(f"  -> Pass 4 complete ({len(final_answer)} chars)")
        print(f"  -> Total improvement: {len(final_answer) - len(initial_answer):+d} chars")
        
        return final_answer
        
    except Exception as e:
        print(f"Groq error: {e}")
        return None

def get_answer(query, session_id):
    """Main routing with conversational memory"""
    
    # 1. Check built-in knowledge
    kb = check_knowledge_base(query)
    if kb:
        return kb
    
    # 2. Check cache
    cached = get_cached_answer(query, session_id)
    if cached:
        cached['cached'] = True
        add_to_history(session_id, query, cached['answer'])
        return cached
    
    # 3. Use Groq 4-pass system
    print(f"  -> Using 4-pass conversational reasoning...")
    answer = groq_four_pass(query, session_id)
    
    if answer:
        result = {
            'answer': answer,
            'sources': [{'title': 'Powered by Groq AI (4-pass conversational)', 'url': 'https://groq.com'}],
            'method': 'Groq 4-Pass Conversational',
            'cached': False,
            'passes': 4
        }
        
        add_to_history(session_id, query, answer)
        save_to_cache(query, session_id, result)
        return result
    
    return {
        'answer': f"AI temporarily unavailable. Please try again.",
        'sources': [],
        'method': 'Error',
        'cached': False
    }

@app.route('/')
def index():
    get_session_id()
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query'}), 400
        
        session_id = get_session_id()
        history_count = len(get_conversation_history(session_id))
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Query: {query} (Session: {session_id[:8]}..., History: {history_count})")
        result = get_answer(query, session_id)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Method: {result.get('method')} | Cached: {result.get('cached')}")
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result.get('sources', []),
            'method': result.get('method'),
            'cached': result.get('cached', False),
            'conversation_length': history_count + 1
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/restart', methods=['POST'])
def restart():
    """Clear conversation history"""
    session_id = get_session_id()
    if session_id in conversation_history:
        old_count = len(conversation_history[session_id])
        conversation_history[session_id].clear()
        return jsonify({
            'success': True,
            'message': f'Conversation restarted. Cleared {old_count} exchanges.',
            'conversation_length': 0
        })
    return jsonify({
        'success': True,
        'message': 'No history to clear.',
        'conversation_length': 0
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    cache_files = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]) if os.path.exists(CACHE_DIR) else 0
    
    total_queries = request_counter['cache_hits'] + request_counter['cache_misses']
    cache_hit_rate = f"{(request_counter['cache_hits'] / total_queries * 100):.1f}%" if total_queries > 0 else "0%"
    
    session_id = get_session_id()
    current_history_length = len(get_conversation_history(session_id))
    
    return jsonify({
        'api_requests_made': request_counter['count'],
        'cached_answers': cache_files,
        'cache_hits': request_counter['cache_hits'],
        'cache_misses': request_counter['cache_misses'],
        'cache_hit_rate': cache_hit_rate,
        'total_queries': total_queries,
        'active_conversations': len(conversation_history),
        'current_conversation_length': current_history_length,
        'rate_limit_hits': request_counter['rate_limit_hits']
    })

if __name__ == '__main__':
    print("=" * 60)
    print("FRANKENAI v9.0 - PRODUCTION READY")
    print("=" * 60)
    print("URL: http://localhost:5000")
    print("=" * 60)
    print("FEATURES:")
    print("  - 4-pass conversational reasoning")
    print("  - Exponential backoff retry logic")
    print("  - Optimized token usage (~30% reduction)")
    print("  - Environment variable API key")
    print("  - Conversation memory (last 3 exchanges)")
    print("  - Smart caching with context")
    print("  - Rate limit tracking")
    print("=" * 60)
    print(f"API Key: {'✓ Loaded' if GROQ_API_KEY else '✗ Missing'}")
    print(f"Cache: {os.path.abspath(CACHE_DIR)}")
    print(f"Stats: http://localhost:5000/api/stats")
    print(f"Restart: POST to http://localhost:5000/api/restart")
    print("=" * 60)
    app.run(debug=True, port=5000, host='0.0.0.0')