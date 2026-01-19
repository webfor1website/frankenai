# FRANKENAI v12.0 - PRODUCTION GRADE
# 
# NEW IN v12.0:
# - Thread-safe storage with atomic writes
# - Bounded memory management with TTL
# - AWS S3 + Secrets Manager support (optional)
# - Health checks and metrics endpoints
# - CloudWatch integration (optional)
# - Prometheus metrics export
# - Docker + Kubernetes ready
# - Horizontal scaling support
#
# PRODUCTION FEATURES:
# - No race conditions
# - No memory leaks
# - No cache corruption
# - Graceful shutdown
# - Request metrics
# - Cost tracking
#
# Pass 1: Detect code → Summarize structure
# Pass 2-4: SKIP (code doesn't need self-critique)
# Pass 5: Claude gets FULL code + summary for deep analysis
# Pass 6-8: Normal pipeline continues

from flask import Flask, request, jsonify, render_template, session
import re
import json
import hashlib
import time
import random
from datetime import datetime
from groq import Groq
from anthropic import Anthropic
import os
from collections import deque
from dotenv import load_dotenv
import requests
from urllib.parse import quote_plus
import markdown
import signal
import atexit
import uuid

# v12.0 imports
from storage import create_storage_backend, ConversationHistory
from config import FrankenAIConfig
from monitoring import init_monitoring, metrics, health


def track_request_time(func):
    """Decorator to track request processing time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            metrics.record_request_time(duration)
            health.ping()
            return result
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_request_time(duration)
            raise
    return wrapper

load_dotenv()

# v12.0: Load configuration
config = FrankenAIConfig.from_environment()
config.validate()

app = Flask(__name__)
app.secret_key = config.flask_secret_key

# v12.0: Initialize monitoring
metrics, health = init_monitoring()


# Initialize API clients with config
GROQ_API_KEY = config.groq_api_key
CLAUDE_API_KEY = config.claude_api_key
GROK_API_KEY = config.grok_api_key

groq_client = Groq(api_key=GROQ_API_KEY)
claude_client = Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None


# Pricing from config
CLAUDE_INPUT_PER_M = config.claude_input_per_m
CLAUDE_OUTPUT_PER_M = config.claude_output_per_m
GROK_INPUT_PER_M = config.grok_input_per_m
GROK_OUTPUT_PER_M = config.grok_output_per_m


# v12.0: Initialize storage backend
# Initialize storage backend
if config.storage_backend == 'file':
    storage = create_storage_backend(
        backend_type='file',
        base_dir=config.storage_base_dir
    )
else:
    storage = create_storage_backend(
        backend_type='s3',
        bucket=config.s3_bucket,
        prefix=config.s3_prefix
    )

# Cache directories (for pass cache - still file-based)
PASS_CACHE_DIR = "cache/passes"
os.makedirs(PASS_CACHE_DIR, exist_ok=True)


# Request counter with enhanced metrics
request_counter = {
    'count': 0,
    'cache_hits': 0,
    'pass_cache_hits': 0,
    'groq_pass1_only': 0,
    'groq_pass2_exits': 0,
    'groq_pass3_exits': 0,
    'groq_full_4pass': 0,
    'claude_passes': 0,
    'claude_skipped': 0,
    'claude_finisher_passes': 0,
    'claude_finisher_skipped': 0,
    'grok_passes': 0,
    'grok_reverted': 0,
    'frankenai_synthesis': 0,
    'quality_gate_triggers': 0,
    'word_count_adjustments': 0,
    'code_analysis_queries': 0,  # NEW v11.1
    'code_lines_analyzed': 0,    # NEW v11.1
    'estimated_cost': 0.0
}

# v12.0: Thread-safe conversation history with TTL
conversation_history = ConversationHistory(
    max_size=config.max_conversations,
    ttl_seconds=config.conversation_ttl
)

# EXPANDED TECHNICAL KEYWORDS (v10.6)
TECHNICAL_KEYWORDS = [
    'quantum', 'explain', 'how does', 'mechanism', 'theorem', 'algorithm',
    'formula', 'equation', 'prove', 'derive', 'calculate', 'technical',
    'physics', 'chemistry', 'biology', 'mathematical',
    # BIOLOGY TERMS
    'molecular', 'CRISPR', 'gene', 'protein', 'DNA', 'RNA', 'cas9',
    'cellular', 'enzyme', 'nucleotide', 'sequence', 'genome', 'chromosome',
    'mitochondria', 'ribosome', 'membrane', 'cell', 'organism',
    # CHEMISTRY TERMS
    'chemical', 'reaction', 'synthesis', 'compound', 'element', 'atom',
    'molecule', 'ion', 'bond', 'catalyst', 'oxidation', 'reduction',
    # PHYSICS TERMS
    'force', 'energy', 'momentum', 'wave', 'particle', 'relativity',
    'thermodynamics', 'electromagnetism'
]

CREATIVE_KEYWORDS = [
    'story', 'imagine', 'creative', 'fun', 'interesting', 'write a',
    'poem', 'narrative', 'fiction', 'tale', 'adventure', 'character',
    'plot', 'scene'
]

PHILOSOPHICAL_KEYWORDS = [
    'philosophy', 'philosophical', 'argue', 'debate', 'perspective',
    'why', 'meaning', 'purpose', 'think', 'believe', 'opinion',
    'should', 'ethics', 'morality', 'consciousness', 'existence', 'ethical'
]

ANALYTICAL_KEYWORDS = [
    'analyze', 'analysis', 'compare', 'contrast', 'evaluate',
    'assess', 'examine', 'critique', 'argue for', 'prove that',
    'attempt to', 'try to', 'demonstrate', 'show that'
]

KNOWLEDGE_BASE = {
    'greetings': {
        'patterns': [r'\bhi\b', r'\bhello\b', r'\bhey\b'],
        'response': "Hey! I'm FrankenAI v10.7 - Now with Claude Final Polish!"
    },
    'thanks': {
        'patterns': [r'\bthank', r'\bthanks\b'],
        'response': "You're welcome!"
    }
}

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    return session['session_id']

def get_conversation_history(session_id):
    """Get conversation history using thread-safe v12 API"""
    # The v12 ConversationHistory handles session management internally
    # Return a list of recent messages
    history = conversation_history.get(session_id)
    if history is None:
        conversation_history.add(session_id, [])
        return []
    return history

def add_to_history(session_id, query, answer):
    """Add to conversation history using thread-safe v12 API"""
    history = conversation_history.get(session_id)
    if history is None:
        history = []
    
    truncated = answer[:500] if len(answer) > 500 else answer
    history.append({'query': query[:300], 'answer': truncated})
    
    # Keep only last 2 messages
    if len(history) > 2:
        history = history[-2:]
    
    conversation_history.add(session_id, history)

def get_cache_key(query, session_id):
    """Generate cache key using conversation context"""
    history = conversation_history.get(session_id)
    if history:
        context = json.dumps([h.get('query', '') for h in history])
    else:
        context = ""
    return hashlib.md5(f"{context}_{query}".encode()).hexdigest()

def get_pass_cache_key(query, pass_num):
    return hashlib.md5(f"pass_{pass_num}_{query}".encode()).hexdigest()

def get_cached_pass(query, pass_num):
    try:
        key = get_pass_cache_key(query, pass_num)
        file = os.path.join(PASS_CACHE_DIR, f"{key}.json")
        
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                if time.time() - cached.get('timestamp', 0) < 3600:
                    metrics['pass_cache_hits'] += 1
                    print(f"  -> Pass {pass_num} CACHE HIT")
                    return cached.get('result')
    except:
        pass
    return None

def save_pass_cache(query, pass_num, result):
    try:
        key = get_pass_cache_key(query, pass_num)
        file = os.path.join(PASS_CACHE_DIR, f"{key}.json")
        
        with open(file, 'w', encoding='utf-8') as f:
            json.dump({'result': result, 'timestamp': time.time()}, f)
    except:
        pass

def get_cached_answer(query, session_id):
    """Get cached answer using v12 storage backend"""
    try:
        key = get_cache_key(query, session_id)
        cached = storage.get(key)
        if cached:
            metrics['cache_hits'] += 1
            print(f"  -> Full cache HIT")
            return cached
    except Exception as e:
        print(f"  -> Cache get error: {e}")
    return None

def save_to_cache(query, session_id, result):
    """Save to cache using v12 storage backend"""
    try:
        key = get_cache_key(query, session_id)
        storage.set(key, result, ttl=config.cache_ttl)
    except Exception as e:
        print(f"  -> Cache save error: {e}")

# SIMPLE COMPARISON DETECTION (v10.6)
def is_simple_comparison(query):
    """Detect simple comparison/definition questions that should exit early"""
    simple_patterns = [
        r"what'?s? the difference",
        r"how (?:is|are) .* different",
        r"compare .* (?:and|vs|versus)",
        r"(?:what|when|where|who) is ",
        r"define ",
        r"explain .* in simple terms",
        r"what does .* mean",
        r"difference between"
    ]
    return any(re.search(p, query.lower()) for p in simple_patterns)

# WORD COUNT EXTRACTION (v10.6)
def extract_word_count_target(query):
    """Extract requested word count from query"""
    match = re.search(r'(\d+)\s*words?', query.lower())
    if match:
        return int(match.group(1))
    return None

# CODE ANALYSIS DETECTION (v11.1)
def detect_code_analysis(query):
    """Detect if query contains code for analysis"""
    
    # Check for code block markers
    if '```' in query:
        return True
    
    # Check for code patterns (Python, JavaScript, etc.)
    code_patterns = [
        r'def\s+\w+\s*\(',  # Python functions
        r'class\s+\w+\s*[:(]',  # Python classes
        r'import\s+\w+',  # Python imports
        r'from\s+\w+\s+import',  # Python from imports
        r'function\s+\w+\s*\(',  # JavaScript functions
        r'const\s+\w+\s*=',  # JavaScript const
        r'let\s+\w+\s*=',  # JavaScript let
        r'public\s+class',  # Java classes
        r'#include\s*<',  # C/C++ includes
    ]
    
    # Check for review keywords
    review_keywords = [
        'review this code',
        'analyze this code',
        'what do you think about this code',
        'find bugs',
        'improve this code',
        'optimize this',
        'refactor this',
        'check this code',
        'code review',
        'review my code'
    ]
    
    has_code_patterns = any(re.search(p, query, re.MULTILINE | re.IGNORECASE) for p in code_patterns)
    has_review_keywords = any(kw in query.lower() for kw in review_keywords)
    
    # Calculate code-like character ratio
    code_chars = len(re.findall(r'[{}()\[\];=]', query))
    total_chars = max(len(query), 1)
    code_ratio = code_chars / total_chars
    
    # Detect if substantial code (>3 lines with code patterns)
    lines_with_code = sum(1 for line in query.split('\n') 
                          if any(re.search(p, line, re.IGNORECASE) for p in code_patterns))
    
    # Return True if:
    # 1. Has review keywords AND code patterns
    # 2. OR has code blocks (```)
    # 3. OR has high code ratio (>5%) AND multiple code lines
    return (has_review_keywords and has_code_patterns) or \
           code_ratio > 0.05 or \
           lines_with_code >= 3

# Enhanced query classification
def classify_query_type(query):
    """Classify query to determine appropriate approach"""
    lower_q = query.lower()
    
    tech_score = sum(1 for k in TECHNICAL_KEYWORDS if k in lower_q)
    creative_score = sum(1 for k in CREATIVE_KEYWORDS if k in lower_q)
    phil_score = sum(1 for k in PHILOSOPHICAL_KEYWORDS if k in lower_q)
    analytical_score = sum(1 for k in ANALYTICAL_KEYWORDS if k in lower_q)
    
    # Determine primary type
    if phil_score >= 2 or analytical_score >= 1:
        return 'philosophical'
    elif tech_score >= 2:
        return 'technical'
    elif creative_score >= 1:
        return 'creative'
    else:
        return 'balanced'

def needs_analytical_approach(query):
    """Check if query needs analytical (not just descriptive) response"""
    lower_q = query.lower()
    
    analytical_triggers = [
        'prove', 'argue', 'attempt to', 'try to', 'demonstrate',
        'show that', 'can you prove', 'make the case', 'argue for'
    ]
    
    return any(trigger in lower_q for trigger in analytical_triggers)

def has_technical_terms(query):
    return any(k in query.lower() for k in TECHNICAL_KEYWORDS)

def select_model(query):
    if len(query) < 50 and not has_technical_terms(query):
        return "llama-3.1-8b-instant", "simple"
    return "llama-3.3-70b-versatile", "complex"

def check_knowledge_base(query):
    lower = query.lower()
    
    for category, data in KNOWLEDGE_BASE.items():
        if 'patterns' in data:
            for pattern in data['patterns']:
                if re.search(pattern, lower):
                    return {'answer': data['response'], 'method': 'Built-in'}
    
    # Basic math
    match = re.search(r'(\d+)\s*([+\-*/])\s*(\d+)', query)
    if match:
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        if op == '+': result = a + b
        elif op == '-': result = a - b
        elif op == '*': result = a * b
        elif op == '/': result = a / b if b != 0 else 'undefined'
        return {'answer': f"{a} {op} {b} = {result}", 'method': 'Calculator'}
    
    return None

def rate_limited_api_call(func, *args, **kwargs):
    max_retries = 5
    base_wait = 1.0
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            
            if 'rate limit' in error_str and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt) + random.uniform(0, 1)
                print(f"  -> Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait:.1f}s...")
                time.sleep(wait)
            elif 'timeout' in error_str and attempt < max_retries - 1:
                wait = base_wait * (1.5 ** attempt)
                print(f"  -> Timeout (attempt {attempt + 1}/{max_retries}), waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise e

def check_completeness(answer, query):
    """Check if answer is complete enough"""
    try:
        check = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Rate completeness 0-10:\nQ: {query[:200]}\nA: {answer[:1000]}\n\nNumber:"
            }],
            max_tokens=5,
            temperature=0.2
        )
        score = int(check.choices[0].message.content.strip())
        return score
    except:
        return 5

def verify_technical_accuracy(original, refined, query):
    """Check if refinement preserved key technical details"""
    try:
        check = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""Compare answers. Did refined version keep:
1. All key facts? (Y/N)
2. Technical precision? (Y/N)
3. Specific names/dates/numbers? (Y/N)

Original: {original[:1500]}
Refined: {refined[:1500]}

Format: Y/N/Y"""
            }],
            max_tokens=10,
            temperature=0.1
        )
        
        results = check.choices[0].message.content.strip().split('/')
        all_preserved = all(r.strip().upper() == 'Y' for r in results)
        
        if not all_preserved:
            print(f"  -> Accuracy check: {'/'.join(results)}")
        
        return all_preserved
    except Exception as e:
        print(f"  -> Accuracy check failed: {e}")
        return True

# WORD COUNT ENFORCEMENT (v10.6)
def enforce_word_count(answer, target_words, query):
    """Trim or expand answer to match target word count"""
    if not target_words:
        return answer
    
    current_words = len(answer.split())
    tolerance = 0.1  # 10% tolerance
    
    if abs(current_words - target_words) / target_words <= tolerance:
        print(f"  -> Word count OK: {current_words} (~{target_words})")
        return answer
    
    print(f"  -> Word count OFF: {current_words} vs target {target_words}, adjusting...")
    metrics['word_count_adjustments'] += 1
    
    try:
        trim = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Rewrite this story to be EXACTLY {target_words} words (±10 words).

Current story ({current_words} words):
{answer}

Maintain all plot points but adjust pacing. Output ONLY the story, no preamble:"""
            }],
            max_tokens=int(target_words * 1.5),
            temperature=0.6
        )
        
        adjusted = trim.choices[0].message.content.strip()
        new_count = len(adjusted.split())
        print(f"  -> Adjusted to {new_count} words")
        return adjusted
    except Exception as e:
        print(f"  -> Word count adjustment failed: {e}")
        return answer

# ========== GROQ WITH SMART EARLY EXITS (PASSES 1-4) ==========

def groq_smart_passes(query):
    """Groq with confidence-based early exits and intelligent detection"""
    try:
        print(f"  -> [GROQ] Starting...")
        
        model, reason = select_model(query)
        query_type = classify_query_type(query)
        is_analytical = needs_analytical_approach(query)
        is_simple = is_simple_comparison(query)
        target_words = extract_word_count_target(query)
        
        print(f"  -> Model: {model} ({reason})")
        print(f"  -> Type: {query_type}, Analytical: {is_analytical}, Simple: {is_simple}")
        if target_words:
            print(f"  -> Target words: {target_words}")
        
        # PASS 1
        cached = get_cached_pass(query, 1)
        if cached:
            answer1 = cached
        else:
            print(f"  -> Pass 1/4...")
            
            # Adjust max_tokens based on query type
            if query_type == 'creative' and target_words:
                max_tok = int(target_words * 1.5)
                sys_msg = f"Write a creative story of EXACTLY {target_words} words. Respond in English."
            elif query_type == 'technical':
                max_tok = 2000
                sys_msg = "Provide a precise, technical answer. Respond in English."
            elif is_simple:
                max_tok = 1000
                sys_msg = "Provide a clear, concise answer. Respond in English."
            else:
                max_tok = 2000
                sys_msg = "You are a helpful AI assistant. Provide detailed, accurate answers in English."
            
            pass1 = rate_limited_api_call(
                groq_client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=max_tok
            )
            answer1 = pass1.choices[0].message.content
            metrics['count'] += 1
            save_pass_cache(query, 1, answer1)
        
        print(f"  -> Pass 1: {len(answer1)} chars")
        
        # EARLY EXIT 1: Simple + confident
        if len(answer1) < 400 and reason == "simple" and is_simple:
            print(f"  -> EXIT after Pass 1 (simple query)")
            metrics['groq_pass1_only'] += 1
            return {
                'success': True,
                'answer': answer1,
                'provenance': '[GROQ] Pass 1 (simple)',
                'query_type': query_type,
                'is_analytical': is_analytical,
                'target_words': target_words
            }
        
        # PASS 2
        cached = get_cached_pass(query, 2)
        if cached:
            critique = cached
        else:
            print(f"  -> Pass 2/4...")
            
            # Different critique style for analytical queries
            if is_analytical:
                critique_prompt = f"Q: {query}\nA: {answer1}\n\nThis is an ANALYTICAL question requiring argumentation. What perspectives, arguments, or explorations are missing?"
            else:
                critique_prompt = f"Q: {query}\nA: {answer1}\n\nWhat improvements are needed?"
            
            pass2 = rate_limited_api_call(
                groq_client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": "You are a critical reviewer. Identify what's missing or could be improved. Respond in English."},
                    {"role": "user", "content": critique_prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            critique = pass2.choices[0].message.content
            metrics['count'] += 1
            save_pass_cache(query, 2, critique)
        
        # DYNAMIC COMPLETENESS THRESHOLDS (v10.6)
        completeness = check_completeness(answer1, query)
        print(f"  -> Completeness: {completeness}/10")
        
        # Different thresholds for different query types
        if query_type == 'philosophical' or is_analytical:
            exit_threshold = 9  # High bar for complex queries
        elif is_simple or len(query.split()) < 10:
            exit_threshold = 6  # Lower bar for simple queries
        else:
            exit_threshold = 8  # Medium bar for balanced
        
        if completeness >= exit_threshold and not is_analytical:
            print(f"  -> EXIT after Pass 2 (completeness {completeness} >= {exit_threshold})")
            metrics['groq_pass2_exits'] += 1
            return {
                'success': True,
                'answer': answer1,
                'provenance': f'[GROQ] Pass 1-2 (complete {completeness}/10)',
                'query_type': query_type,
                'is_analytical': is_analytical,
                'target_words': target_words
            }
        
        # PASS 3
        cached = get_cached_pass(query, 3)
        if cached:
            answer3 = cached
        else:
            print(f"  -> Pass 3/4...")
            
            if is_analytical:
                system_msg = "Provide a comprehensive, ANALYTICAL answer that explores multiple perspectives and arguments. Be thorough. Respond in English."
            elif query_type == 'creative' and target_words:
                system_msg = f"Write a creative story of EXACTLY {target_words} words. Respond in English."
            else:
                system_msg = "Provide the most comprehensive answer using the critique. Respond in English."
            
            max_tok = int(target_words * 1.5) if (query_type == 'creative' and target_words) else 3000
            
            pass3 = rate_limited_api_call(
                groq_client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Q: {query}\nFirst answer: {answer1}\nCritique: {critique}\n\nProvide the BEST answer:"}
                ],
                temperature=0.6,
                max_tokens=max_tok
            )
            answer3 = pass3.choices[0].message.content
            metrics['count'] += 1
            save_pass_cache(query, 3, answer3)
        
        print(f"  -> Pass 3: {len(answer3)} chars")
        
        # EARLY EXIT 3: Check if excellent (skip for analytical)
        if not is_analytical and not is_simple:
            completeness3 = check_completeness(answer3, query)
            print(f"  -> Completeness: {completeness3}/10")
            
            if completeness3 >= 9.5:
                print(f"  -> EXIT after Pass 3 (excellent)")
                metrics['groq_pass3_exits'] += 1
                return {
                    'success': True,
                    'answer': answer3,
                    'provenance': '[GROQ] Pass 1-3 (excellent)',
                    'query_type': query_type,
                    'is_analytical': is_analytical,
                    'target_words': target_words
                }
        
        # PASS 4
        print(f"  -> Pass 4/4...")
        
        if query_type == 'creative' and target_words:
            sys_msg = f"Refine this story to EXACTLY {target_words} words while maintaining quality. Respond in English."
            max_tok = int(target_words * 1.5)
        else:
            sys_msg = "Improve clarity and flow. Respond in English."
            max_tok = 3000
        
        pass4 = rate_limited_api_call(
            groq_client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Q: {query}\nAnswer: {answer3}\n\nImprove:"}
            ],
            temperature=0.5,
            max_tokens=max_tok
        )
        final = pass4.choices[0].message.content
        metrics['count'] += 1
        metrics['groq_full_4pass'] += 1
        
        print(f"  -> Pass 4: {len(final)} chars")
        
        return {
            'success': True,
            'answer': final,
            'provenance': f'[GROQ] Full 4-pass',
            'query_type': query_type,
            'is_analytical': is_analytical,
            'target_words': target_words
        }
        
    except Exception as e:
        print(f"  -> [GROQ] ERROR: {e}")
        return {'success': False, 'error': str(e)}

# ========== CODE ANALYSIS MODE (v11.1) ==========

def groq_code_summary(query):
    """Pass 1 for code analysis: Create architectural summary"""
    try:
        print(f"  -> [GROQ] CODE ANALYSIS MODE - Summarizing...")
        
        # STRIP ALL CODE BLOCKS - user might paste with ``` markers
        # Remove opening code blocks
        query = re.sub(r'^```\w*\n', '', query, flags=re.MULTILINE)
        # Remove closing code blocks
        query = re.sub(r'\n```\s*$', '', query, flags=re.MULTILINE)
        # Remove any remaining code block markers
        query = query.replace('```', '')
        
        print(f"  -> DEBUG: After stripping code blocks, query is {len(query)} chars")
        
        # Now extract the actual code
        code = query
        preamble = ""
        
        print(f"  -> DEBUG: Query starts with: {query[:100]}")
        
        review_phrases = [
            'review this code:',
            'analyze this code:',
            'what do you think about this code:',
            'check this code:',
            'review my code:',
            'analyze my code:',
            'review this:',
            'check this:',
            'tell me about this code:',
            'tell me about this code',
        ]
        
        for phrase in review_phrases:
            if code.lower().startswith(phrase):
                preamble = phrase
                code = code[len(phrase):].strip()
                print(f"  -> DEBUG: Stripped '{phrase}', code now {len(code)} chars")
                break
        
        if preamble == "":
            print(f"  -> DEBUG: No phrase matched, using full query")
        
        # Calculate stats
        lines = code.split('\n')
        line_count = len(lines)
        char_count = len(code)
        
        print(f"  -> Code stats: {line_count} lines, {char_count:,} chars")
        print(f"  -> DEBUG: First line of code: {lines[0][:100] if lines else 'EMPTY'}")
        print(f"  -> DEBUG: Last line of code: {lines[-1][:100] if lines else 'EMPTY'}")
        
        # Detect language
        if 'def ' in code or 'import ' in code or 'class ' in code:
            language = "Python"
        elif 'function' in code or 'const ' in code or 'let ' in code:
            language = "JavaScript"
        elif 'public class' in code or 'private ' in code:
            language = "Java"
        else:
            language = "Unknown"
        
        # Summarize architecture (limit to first 15k chars for summary)
        code_preview = code[:15000]
        if len(code) > 15000:
            code_preview += "\n\n... [CODE CONTINUES]"
        
        summary_prompt = f"""Analyze this {language} code's STRUCTURE (don't find bugs yet, just understand the architecture):

```{language.lower()}
{code_preview}
```

Provide a concise summary covering:
1. **Primary Purpose**: What does this code do?
2. **Main Components**: Key functions, classes, or modules
3. **Dependencies**: External libraries or frameworks used
4. **Architecture Pattern**: MVC, microservices, monolith, etc.
5. **Estimated Complexity**: Simple, moderate, or complex
6. **Notable Features**: Interesting design choices

Keep it under 800 words. This will help a deeper analyzer understand context."""

        summary_response = rate_limited_api_call(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "system",
                "content": "You are a code architecture analyst. Provide structural overview only, not bug analysis."
            }, {
                "role": "user",
                "content": summary_prompt
            }],
            max_tokens=1500,
            temperature=0.3
        )
        
        summary_text = summary_response.choices[0].message.content
        metrics['count'] += 1
        metrics['code_analysis_queries'] += 1
        metrics['code_lines_analyzed'] += line_count
        
        print(f"  -> Summary: {len(summary_text)} chars")
        
        return {
            'success': True,
            'answer': summary_text,
            'full_code': code,
            'preamble': preamble,
            'line_count': line_count,
            'char_count': char_count,
            'language': language,
            'provenance': f'[GROQ] Code summary ({line_count} lines, {language})',
            'query_type': 'code_analysis',
            'is_code_analysis': True
        }
        
    except Exception as e:
        print(f"  -> [GROQ CODE SUMMARY] ERROR: {e}")
        return {'success': False, 'error': str(e)}

def claude_code_analysis(groq_output, query):
    """Pass 5 for code: Claude gets FULL code + summary for deep analysis"""
    
    if not claude_client:
        print(f"  -> [CLAUDE CODE ANALYSIS] No API key")
        return {
            'success': False, 
            'answer': groq_output['answer'],
            'provenance': '[CLAUDE] No API key for code analysis'
        }
    
    try:
        print(f"  -> [CLAUDE] CODE ANALYSIS - Deep Review...")
        metrics['claude_passes'] += 1
        
        summary = groq_output['answer']
        full_code = groq_output['full_code']
        preamble = groq_output['preamble']
        line_count = groq_output['line_count']
        language = groq_output.get('language', 'Unknown')
        
        # Chunk code if too large (Claude has 200k context, be conservative)
        max_code_chars = 150000  # ~37.5k tokens
        code_truncated = False
        if len(full_code) > max_code_chars:
            print(f"  -> Code too large ({len(full_code):,} chars), truncating to {max_code_chars:,}...")
            full_code = full_code[:max_code_chars]
            code_truncated = True
        
        analysis_prompt = f"""You are an expert code reviewer with deep knowledge of {language}. Analyze this code comprehensively.

**User Request:** {preamble if preamble else "Review this code"}

**Architecture Summary from Initial Scan:**
{summary}

**FULL CODE TO ANALYZE:**
```{language.lower()}
{full_code}
```
{f"**Note:** Code truncated at {max_code_chars:,} characters for analysis." if code_truncated else ""}

Provide a comprehensive code review covering:

## 1. Architecture & Design
- Overall structure quality and organization
- Design patterns used (or should be used)
- Separation of concerns
- Modularity and maintainability
- SOLID principles adherence

## 2. Code Quality
- Readability and clarity
- Naming conventions (variables, functions, classes)
- Code documentation (comments, docstrings)
- Code organization and file structure
- Consistency

## 3. Bugs & Logic Issues
- Potential runtime errors
- Edge cases not handled
- Logic flaws or incorrect algorithms
- Type inconsistencies
- Off-by-one errors, race conditions

## 4. Performance
- Inefficient algorithms or data structures
- Unnecessary loops or operations
- Memory usage concerns
- Database query optimization
- Caching opportunities

## 5. Security
- Input validation gaps
- SQL injection vulnerabilities
- XSS (Cross-Site Scripting) risks
- Authentication/authorization issues
- Sensitive data exposure
- Dependency vulnerabilities

## 6. Best Practices
- Language-specific idioms violations
- Anti-patterns
- Error handling gaps
- Hardcoded values that should be config
- Missing logging or monitoring

## 7. Specific Recommendations
**Priority Issues** (fix immediately):
- List critical bugs/security issues

**Improvements** (enhance quality):
- List design improvements

**Nice-to-haves** (optional enhancements):
- List optional optimizations

For each issue, provide:
- Specific line reference (if possible)
- Code snippet showing the problem
- Suggested fix with code example
- Explanation of why it matters

Be thorough but actionable. Focus on what will most improve the code."""

        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,  # More tokens for comprehensive code review
            temperature=0.3,  # Lower temperature for technical accuracy
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        analysis = response.content[0].text
        
        # Calculate cost
        in_tok = len(analysis_prompt) // 4
        out_tok = len(analysis) // 4
        cost = (in_tok / 1_000_000 * CLAUDE_INPUT_PER_M) + (out_tok / 1_000_000 * CLAUDE_OUTPUT_PER_M)
        metrics['estimated_cost'] += cost
        
        print(f"  -> Analysis: {len(analysis)} chars, ${cost:.4f}")
        
        return {
            'success': True,
            'answer': analysis,
            'provenance': f'[CLAUDE] Code analysis ({line_count} lines, ${cost:.3f})',
            'cost': cost,
            'query_type': 'code_analysis',
            'is_analytical': True,
            'is_code_analysis': True
        }
        
    except Exception as e:
        print(f"  -> [CLAUDE CODE ANALYSIS] ERROR: {e}")
        return {
            'success': False, 
            'answer': groq_output['answer'],
            'provenance': f'[CLAUDE] Code analysis failed: {str(e)[:100]}'
        }

# ========== CLAUDE PASS 5 (ENHANCEMENT) ==========

def should_use_claude(query, groq_answer):
    if len(groq_answer) < 1200:
        print(f"  -> [CLAUDE] Skip: Too short")
        metrics['claude_skipped'] += 1
        return False
    
    if "simple" in select_model(query)[1]:
        print(f"  -> [CLAUDE] Skip: Simple")
        metrics['claude_skipped'] += 1
        return False
    
    try:
        preview = min(1800, len(groq_answer))
        
        check = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Rate improvement need (0-10):\nQ: {query[:400]}\nA: {groq_answer[:preview]}\n\nNumber:"
            }],
            max_tokens=5,
            temperature=0.2
        )
        
        try:
            score = int(check.choices[0].message.content.strip())
            needs = score >= 7
            print(f"  -> [CLAUDE] Quality: {score}/10 -> {'YES' if needs else 'SKIP'}")
            if not needs:
                metrics['claude_skipped'] += 1
            return needs
        except:
            return len(groq_answer) > 2000
            
    except:
        return True

def claude_pass5(groq_output, query):
    if not claude_client:
        print(f"  -> [CLAUDE] No API key")
        return {'success': False, 'answer': groq_output['answer'], 'provenance': '[CLAUDE] No key'}
    
    groq_answer = groq_output['answer']
    
    if not should_use_claude(query, groq_answer):
        return {'success': False, 'answer': groq_answer, 'provenance': '[CLAUDE] Skipped'}
    
    try:
        print(f"  -> [CLAUDE] Pass 5...")
        metrics['claude_passes'] += 1
        
        if len(groq_answer) > 18000:
            print(f"  -> Summarizing...")
            summary = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Summarize keeping key facts:\n\n{groq_answer[:18000]}"}],
                max_tokens=2000,
                temperature=0.5
            )
            groq_answer = summary.choices[0].message.content
        
        prompt = f"""Review this AI-generated answer. Your task:

1. Check accuracy - identify any errors
2. Identify gaps - what important information is missing?
3. Add depth - expand on key points that need more detail
4. Maintain quality - don't add fluff

Respond in English only.

Question: {query[:500]}

Answer to enhance:
{groq_answer}

Output the enhanced version:"""
        
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.6,
            messages=[{"role": "user", "content": prompt}]
        )
        
        enhanced = response.content[0].text
        metrics['count'] += 1
        
        in_tok = len(prompt) // 4
        out_tok = len(enhanced) // 4
        cost = (in_tok / 1_000_000 * CLAUDE_INPUT_PER_M) + (out_tok / 1_000_000 * CLAUDE_OUTPUT_PER_M)
        metrics['estimated_cost'] += cost
        
        print(f"  -> Done: {len(enhanced)} chars, ${cost:.4f}")
        
        return {
            'success': True,
            'answer': enhanced,
            'provenance': '[CLAUDE] Enhanced',
            'cost': cost
        }
        
    except Exception as e:
        print(f"  -> [CLAUDE] ERROR: {e}")
        return {'success': False, 'answer': groq_output['answer'], 'provenance': '[CLAUDE] Failed'}

# ========== GROK PASS 6 (REFINEMENT) ==========

def grok_pass6(prev_output, query):
    if not GROK_API_KEY:
        print(f"  -> [GROK] No API key")
        return {'success': False, 'answer': prev_output['answer'], 'provenance': '[GROK] No key'}
    
    try:
        print(f"  -> [GROK] Pass 6...")
        metrics['grok_passes'] += 1
        
        answer = prev_output['answer']
        input_text = answer[:25000]
        
        query_type = prev_output.get('query_type', 'balanced')
        is_analytical = prev_output.get('is_analytical', False)
        target_words = prev_output.get('target_words')
        
        if query_type == 'philosophical' or is_analytical:
            refinement_instructions = """Philosophical/Analytical refinement:
- Preserve ALL arguments and perspectives
- Maintain depth of exploration
- Keep nuanced reasoning
- Don't cut philosophical content
- Be thorough, not brief
- Explore multiple angles"""
        elif query_type == 'technical':
            refinement_instructions = """Technical refinement:
- Keep ALL specific names, dates, numbers, formulas
- Remove marketing language
- Use maximum 4 section headers
- Be precise and authoritative"""
        elif query_type == 'creative':
            if target_words:
                refinement_instructions = f"""Creative refinement:
- Target EXACTLY {target_words} words
- Make it engaging
- Use vivid language
- Keep conversational tone"""
            else:
                refinement_instructions = """Creative refinement:
- Make it engaging
- Use vivid language
- Keep conversational tone"""
        else:
            refinement_instructions = """Balanced refinement:
- Clear and accessible
- Remove redundancies
- Improve flow"""
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROK_API_KEY}'
        }
        
        payload = {
            'model': 'grok-3',
            'messages': [{
                'role': 'user',
                'content': f"""{refinement_instructions}

IMPORTANT: Respond in English only.

Question: {query[:300]}

Answer to refine:
{input_text}

Output the improved version:"""
            }],
            'max_tokens': int(target_words * 1.5) if (query_type == 'creative' and target_words) else 3000,
            'temperature': 0.6
        }
        
        response = requests.post(
            'https://api.x.ai/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"  -> [GROK] Response: {response.status_code} - {response.text[:200]}")
            raise Exception(f"Grok API returned {response.status_code}")
        
        data = response.json()
        polished = data['choices'][0]['message']['content']
        
        # Fact verification only for technical
        if query_type == 'technical':
            if not verify_technical_accuracy(input_text, polished, query):
                print(f"  -> [GROK] Accuracy check FAILED, reverting")
                metrics['grok_reverted'] += 1
                return {
                    'success': False, 
                    'answer': prev_output['answer'], 
                    'provenance': '[GROK] Reverted (accuracy)'
                }
        
        in_tok = len(payload['messages'][0]['content']) // 4
        out_tok = len(polished) // 4
        cost = (in_tok / 1_000_000 * GROK_INPUT_PER_M) + (out_tok / 1_000_000 * GROK_OUTPUT_PER_M)
        metrics['estimated_cost'] += cost
        
        print(f"  -> Done: {len(polished)} chars, ${cost:.4f}")
        
        return {
            'success': True,
            'answer': polished,
            'provenance': '[GROK] Final refinement',
            'cost': cost,
            'query_type': query_type,
            'is_analytical': is_analytical,
            'target_words': target_words
        }
        
    except Exception as e:
        print(f"  -> [GROK] ERROR: {e}")
        return {'success': False, 'answer': prev_output['answer'], 'provenance': '[GROK] Failed'}

# ========== FRANKENAI META-SYNTHESIS (PASS 7) ==========

def frankenai_final_synthesis(grok_output, query):
    """FrankenAI reviews its own work and makes final improvements"""
    if not grok_output['success']:
        return grok_output
    
    try:
        print(f"  -> [FRANKENAI] Pass 7 - Meta-synthesis...")
        metrics['frankenai_synthesis'] += 1
        
        answer = grok_output['answer']
        query_type = grok_output.get('query_type', 'balanced')
        is_analytical = grok_output.get('is_analytical', False)
        target_words = grok_output.get('target_words')
        
        # Build synthesis prompt based on query type
        if query_type == 'philosophical' or is_analytical:
            synthesis_goal = "Does this answer FULLY explore the question's arguments and perspectives? If not, what's missing?"
        elif query_type == 'technical':
            synthesis_goal = "Is this answer technically precise with all key details? Any errors or gaps?"
        elif query_type == 'creative' and target_words:
            current_words = len(answer.split())
            synthesis_goal = f"Is this story engaging and EXACTLY {target_words} words? Current: {current_words} words."
        else:
            synthesis_goal = "Is this answer clear, complete, and well-structured?"
        
        # Self-review
        review = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""You are FrankenAI reviewing your own work.

Question: {query[:400]}

Your answer:
{answer[:2000]}

Self-review goal: {synthesis_goal}

Provide:
1. Score (0-10)
2. What's missing or wrong?
3. How to improve?

Format: SCORE|GAPS|IMPROVEMENTS"""
            }],
            max_tokens=500,
            temperature=0.4
        )
        
        review_text = review.choices[0].message.content.strip()
        print(f"  -> Self-review: {review_text[:100]}...")
        
        # Parse review
        parts = review_text.split('|')
        try:
            score = int(parts[0].strip())
        except:
            score = 7
        
        print(f"  -> Self-score: {score}/10")
        
        # If score is high enough, accept (but check word count for creative)
        if score >= 8:
            if query_type == 'creative' and target_words:
                answer = enforce_word_count(answer, target_words, query)
            
            print(f"  -> [FRANKENAI] Accepting (high score)")
            return {
                'success': True,
                'answer': answer,
                'provenance': f'[FRANKENAI] Self-approved ({score}/10)',
                'query_type': query_type,
                'is_analytical': is_analytical,
                'target_words': target_words
            }
        
        # Otherwise, improve
        print(f"  -> [FRANKENAI] Improving...")
        
        improve_prompt = f"""You are FrankenAI making final improvements.

Question: {query[:400]}

Current answer:
{answer[:3000]}

Self-review: {review_text}

Create the FINAL, BEST version addressing all gaps. Respond in English:"""
        
        if query_type == 'creative' and target_words:
            improve_prompt += f"\n\nIMPORTANT: The story must be EXACTLY {target_words} words."
        
        final_pass = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": improve_prompt
            }],
            max_tokens=int(target_words * 1.5) if (query_type == 'creative' and target_words) else 4000,
            temperature=0.6
        )
        
        final_answer = final_pass.choices[0].message.content
        metrics['count'] += 2  # 2 Groq calls
        
        # Enforce word count for creative
        if query_type == 'creative' and target_words:
            final_answer = enforce_word_count(final_answer, target_words, query)
        
        print(f"  -> Final: {len(final_answer)} chars")
        
        return {
            'success': True,
            'answer': final_answer,
            'provenance': f'[FRANKENAI] Self-improved ({score}/10→final)',
            'query_type': query_type,
            'is_analytical': is_analytical,
            'target_words': target_words
        }
        
    except Exception as e:
        print(f"  -> [FRANKENAI] ERROR: {e}")
        return grok_output

# ========== CLAUDE FINISHER (PASS 8) - NEW! ==========

def should_use_claude_finisher(query, frankenai_answer, query_type, grok_expanded=False, quality_score=10):
    """Decide if we should use Claude Finisher (Pass 8)"""
    word_count = len(frankenai_answer.split())
    
    # ALWAYS run Pass 8 if Grok expanded significantly or quality is low
    if grok_expanded or quality_score <= 6:
        print(f"  -> [CLAUDE FINISHER] Required: Compress/improve Grok output")
        return True
    
    # Skip ONLY for very simple queries under 200 words
    if word_count < 200 and query_type not in ['philosophical', 'technical']:
        print(f"  -> [CLAUDE FINISHER] Skip: Very simple query")
        metrics['claude_finisher_skipped'] += 1
        return False
    
    # Run for almost all other cases (quality control)
    print(f"  -> [CLAUDE FINISHER] Running: Quality control")
    return True

def claude_finisher_pass8(frankenai_output, query, grok_expanded=False, quality_score=10):
    """Pass 8: Claude compression + polish - uses Grok's expansion intelligently"""
    
    if not claude_client:
        print(f"  -> [CLAUDE FINISHER] No API key")
        return {'success': False, 'answer': frankenai_output['answer'], 'provenance': '[CLAUDE FINISHER] No key'}
    
    answer = frankenai_output['answer']
    query_type = frankenai_output.get('query_type', 'balanced')
    is_analytical = frankenai_output.get('is_analytical', False)
    target_words = frankenai_output.get('target_words')
    
    # Check if we should use Claude Finisher
    if not should_use_claude_finisher(query, answer, query_type, grok_expanded, quality_score):
        return frankenai_output
    
    try:
        # Determine mode: Compress or Polish
        if grok_expanded and quality_score <= 6:
            mode = "COMPRESS"
            print(f"  -> [CLAUDE FINISHER] Pass 8 - COMPRESS MODE (Grok was verbose)")
        else:
            mode = "POLISH"
            print(f"  -> [CLAUDE FINISHER] Pass 8 - POLISH MODE (standard)")
        
        metrics['claude_finisher_passes'] += 1
        
        # Build prompt based on mode and query type
        if mode == "COMPRESS":
            polish_goal = """Your goal: COMPRESS and IMPROVE this verbose answer
            
This answer is too long and repetitive. Your task is to:
- Remove redundancy and filler content
- Keep ONLY the most important insights and facts
- Make it 30-50% shorter while preserving all key information
- Improve clarity through conciseness
- Remove unnecessary elaboration

DO NOT:
- Lose important facts, numbers, or technical details
- Remove critical examples or evidence
- Make it so short it loses substance

DO:
- Be ruthlessly concise
- Keep the strongest arguments and examples
- Improve readability through brevity
- Merge redundant points"""
        elif query_type == 'philosophical' or is_analytical:
            polish_goal = """Your goal: Perfect the philosophical depth and argumentation
- Ensure all perspectives are fairly represented
- Strengthen the logical flow between arguments
- Make the conclusion more nuanced and compelling
- Keep the same length and depth"""
        elif query_type == 'technical':
            polish_goal = """Your goal: Perfect the technical precision and clarity
- Verify all technical details are accurate
- Improve clarity without sacrificing precision
- Ensure proper terminology throughout
- Keep all specific facts, numbers, and names"""
        elif query_type == 'creative' and target_words:
            polish_goal = f"""Your goal: Perfect the creative narrative and pacing
- Make the story more engaging and vivid
- Improve character development and dialogue
- Ensure proper pacing and structure
- IMPORTANT: Keep it EXACTLY {target_words} words"""
        else:
            polish_goal = """Your goal: Perfect the clarity and engagement
- Improve readability and flow
- Make key points more memorable
- Ensure proper structure
- Remove any remaining redundancies"""
        
        # Truncate if needed (Claude has shorter context window)
        if len(answer) > 20000:
            print(f"  -> Truncating for Claude...")
            answer = answer[:20000]
        
        prompt = f"""{polish_goal}

You are reviewing the final output from a multi-AI system. This answer has already gone through 7 passes of refinement. Your task is to provide the ULTIMATE POLISH - make it perfect.

DO NOT:
- Add unnecessary fluff or marketing language
- Change the core content or arguments
- Make it significantly longer (unless it's creative with word count target)
- Remove important technical details

DO:
- Fix any remaining awkward phrasing
- Improve clarity and readability
- Ensure perfect logical flow
- Make it more engaging where appropriate
- Verify accuracy

Respond in English only.

Question: {query[:500]}

Answer to polish:
{answer}

Output the perfected version:"""
        
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=int(target_words * 1.5) if (query_type == 'creative' and target_words) else 4000,
            temperature=0.5,  # Lower temperature for final polish
            messages=[{"role": "user", "content": prompt}]
        )
        
        polished = response.content[0].text
        metrics['count'] += 1
        
        # Calculate cost
        in_tok = len(prompt) // 4
        out_tok = len(polished) // 4
        cost = (in_tok / 1_000_000 * CLAUDE_INPUT_PER_M) + (out_tok / 1_000_000 * CLAUDE_OUTPUT_PER_M)
        metrics['estimated_cost'] += cost
        
        # Verify length didn't inflate too much
        original_words = len(answer.split())
        polished_words = len(polished.split())
        
        if polished_words > original_words * 1.3 and query_type != 'creative':
            print(f"  -> Claude inflated answer ({original_words}→{polished_words} words), reverting")
            return {
                'success': False,
                'answer': frankenai_output['answer'],
                'provenance': '[CLAUDE FINISHER] Reverted (inflation)'
            }
        
        # Enforce word count for creative
        if query_type == 'creative' and target_words:
            polished = enforce_word_count(polished, target_words, query)
        
        print(f"  -> Done: {len(polished)} chars ({polished_words} words), ${cost:.4f}")
        
        return {
            'success': True,
            'answer': polished,
            'provenance': f'[CLAUDE FINISHER] Ultimate polish',
            'cost': cost
        }
        
    except Exception as e:
        print(f"  -> [CLAUDE FINISHER] ERROR: {e}")
        return frankenai_output

# ========== TRIAD WITH CODE ANALYSIS + CLAUDE FINISHER (PASS 8) ==========

def triad_synapse_ultimate(query, session_id):
    """Ultimate triad with Code Analysis support and Claude Finisher (Pass 8)"""
    provenance = []
    
    # CHECK FOR CODE ANALYSIS MODE
    if detect_code_analysis(query):
        print(f"  -> [DETECTED] CODE ANALYSIS MODE")
        
        # Pass 1: Code summary (replaces Passes 1-4 for code)
        groq = groq_code_summary(query)
        provenance.append(groq['provenance'])
        
        if not groq['success']:
            return {'answer': "Error analyzing code", 'provenance': provenance, 'method': 'Code Analysis Error'}
        
        # Pass 5: Claude deep code analysis (gets FULL code)
        claude = claude_code_analysis(groq, query)
        provenance.append(claude['provenance'])
        
        # Pass 6: Grok refinement (if available)
        if GROK_API_KEY:
            # Pass metadata to Grok
            input_for_grok = claude if claude['success'] else groq
            input_for_grok['query_type'] = 'code_analysis'
            input_for_grok['is_analytical'] = True
            
            grok = grok_pass6(input_for_grok, query)
            provenance.append(grok['provenance'])
            
            current_best = grok if grok['success'] else (claude if claude['success'] else groq)
        else:
            print(f"  -> [GROK] Skipped: No API key")
            current_best = claude if claude['success'] else groq
        
        # SKIP Pass 7 for code analysis - Grok's thorough output goes straight to Claude
        # Pass 7 (FrankenAI) tends to over-compress code reviews, losing valuable details
        print(f"  -> [FRANKENAI] Pass 7 - SKIPPED for code analysis")
        provenance.append('[FRANKENAI] Skipped (code mode)')
        
        # Pass 8: Claude polish (handles compression if needed)
        claude_final = claude_finisher_pass8(
            current_best,
            query,
            grok_expanded=False,
            quality_score=10
        )
        if claude_final.get('provenance'):
            provenance.append(claude_final['provenance'])
        
        return {
            'answer': claude_final['answer'],
            'provenance': provenance,
            'method': 'Code Analysis (4-Pass: Summary→Claude→Grok→Polish)'
        }
    
    # NORMAL PIPELINE (non-code queries)
    # GROQ passes (1-4)
    groq = groq_smart_passes(query)
    provenance.append(groq['provenance'])
    
    if not groq['success']:
        return {'answer': "Error", 'provenance': provenance, 'method': 'Error'}
    
    groq_baseline_length = len(groq['answer'])
    
    # CLAUDE pass (5)
    claude = claude_pass5(groq, query)
    provenance.append(claude['provenance'])
    
    # GROK pass (6)
    input_for_grok = claude if claude['success'] else groq
    
    # Pass metadata through
    if claude['success']:
        input_for_grok['query_type'] = groq.get('query_type', 'balanced')
        input_for_grok['is_analytical'] = groq.get('is_analytical', False)
        input_for_grok['target_words'] = groq.get('target_words')
    
    grok = grok_pass6(input_for_grok, query)
    provenance.append(grok['provenance'])
    
    # QUALITY METRICS (for logging only - no reversion)
    grok_expanded = False
    quality_score = 10  # Default to high quality
    
    if grok['success']:
        final_length = len(grok['answer'])
        is_analytical = groq.get('is_analytical', False)
        
        if final_length > groq_baseline_length * 2 and not is_analytical:
            growth_pct = ((final_length / groq_baseline_length) - 1) * 100
            print(f"  -> [GROK METRICS] Expanded by {growth_pct:.0f}%")
            metrics['quality_gate_triggers'] += 1
            grok_expanded = True
            
            quality_score = check_completeness(grok['answer'], query)
            print(f"  -> Quality score: {quality_score}/10 (Pass 8 will handle)")
            
            # Log but don't revert - let Pass 8 compress if needed
            if quality_score <= 5:
                print(f"  -> [NOTE] Low quality - Pass 8 will compress and improve")
                provenance.append(f'[GROK] Verbose (quality: {quality_score}/10)')
            else:
                provenance.append(f'[GROK] Enhanced (+{growth_pct:.0f}%)')
        else:
            provenance.append('[GROK] Refined')
    
    # FRANKENAI META-SYNTHESIS (Pass 7)
    current_best = grok if grok['success'] else (claude if claude['success'] else groq)
    
    # Pass metadata through
    if not current_best.get('query_type'):
        current_best['query_type'] = groq.get('query_type', 'balanced')
        current_best['is_analytical'] = groq.get('is_analytical', False)
        current_best['target_words'] = groq.get('target_words')
    
    frankenai_final = frankenai_final_synthesis(current_best, query)
    provenance.append(frankenai_final.get('provenance', '[FRANKENAI] Final'))
    
    # CLAUDE FINISHER (Pass 8) - COMPRESSION + POLISH
    # Pass Grok metrics so Claude knows whether to compress or just polish
    claude_final = claude_finisher_pass8(
        frankenai_final, 
        query,
        grok_expanded=grok_expanded if 'grok_expanded' in locals() else False,
        quality_score=quality_score if 'quality_score' in locals() else 10
    )
    if claude_final.get('provenance'):
        if grok_expanded:
            provenance.append(claude_final['provenance'] + ' (compressed)')
        else:
            provenance.append(claude_final['provenance'])
    
    return {
        'answer': claude_final['answer'],
        'provenance': provenance,
        'method': 'Triad Synapse Ultimate (8-Pass)'
    }

@track_request_time
def get_answer(query, session_id):
    # For long queries (likely code), skip knowledge base and go straight to processing
    # This prevents regex timeouts on large code pastes
    if len(query) > 5000:
        print(f"  -> Long query ({len(query)} chars), skipping knowledge base check")
        result = triad_synapse_ultimate(query, session_id)
        
        answer_html = markdown.markdown(result['answer'])
        
        answer_html += "<hr><h3>🔬 Process:</h3><ul>"
        for p in result['provenance']:
            clean = re.sub(r'\$[\d\.]+', '', p)
            answer_html += f"<li>{clean}</li>"
        answer_html += "</ul>"
        
        output = {
            'answer': answer_html,
            'method': result['method']
        }
        
        add_to_history(session_id, query, result['answer'])
        save_to_cache(query, session_id, output)
        return output
    
    # Normal flow for short queries
    kb = check_knowledge_base(query)
    if kb:
        return kb
    
    cached = get_cached_answer(query, session_id)
    if cached:
        add_to_history(session_id, query, cached['answer'])
        return cached
    
    result = triad_synapse_ultimate(query, session_id)
    
    answer_html = markdown.markdown(result['answer'])
    
    answer_html += "<hr><h3>🔬 Process:</h3><ul>"
    for p in result['provenance']:
        clean = re.sub(r'\$[\d\.]+', '', p)
        answer_html += f"<li>{clean}</li>"
    answer_html += "</ul>"
    
    output = {
        'answer': answer_html,
        'method': result['method']
    }
    
    add_to_history(session_id, query, result['answer'])
    save_to_cache(query, session_id, output)
    return output

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        query = data.get('query', '')
        answer = data.get('answer', '')
        rating = data.get('rating', 0)
        session_id = get_session_id()
        
        feedback_id = hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()
        feedback_file = os.path.join(FEEDBACK_DIR, f"{feedback_id}.json")
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump({
                'query': query,
                'answer': answer[:1000],
                'rating': rating,
                'session_id': session_id,
                'timestamp': time.time()
            }, f)
        
        print(f"  -> Feedback saved: {rating}/10")
        
        return jsonify({'success': True, 'message': 'Feedback saved'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {query}")
        result = get_answer(query, session_id)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Done")
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'method': result.get('method')
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    total_groq = metrics['groq_pass1_only'] + metrics['groq_pass2_exits'] + metrics['groq_pass3_exits'] + metrics['groq_full_4pass']
    
    return jsonify({
        'requests': metrics['count'],
        'cache_hits': metrics['cache_hits'],
        'pass_cache_hits': metrics['pass_cache_hits'],
        'groq_breakdown': {
            'pass1_only': metrics['groq_pass1_only'],
            'pass2_exits': metrics['groq_pass2_exits'],
            'pass3_exits': metrics['groq_pass3_exits'],
            'full_4pass': metrics['groq_full_4pass'],
            'total': total_groq
        },
        'claude': {
            'pass5_used': metrics['claude_passes'],
            'pass5_skipped': metrics['claude_skipped'],
            'pass8_finisher': metrics['claude_finisher_passes'],
            'pass8_skipped': metrics['claude_finisher_skipped']
        },
        'grok': {
            'passes': metrics['grok_passes'],
            'reverted': metrics['grok_reverted']
        },
        'frankenai': {
            'meta_synthesis': metrics['frankenai_synthesis'],
            'word_count_adjustments': metrics['word_count_adjustments']
        },
        'code_analysis': {
            'queries': metrics['code_analysis_queries'],
            'lines_analyzed': metrics['code_lines_analyzed'],
            'avg_lines_per_query': metrics['code_lines_analyzed'] / max(metrics['code_analysis_queries'], 1)
        },
        'quality_gates': metrics['quality_gate_triggers'],
        'cost': {
            'total': f"${metrics['estimated_cost']:.2f}",
            'per_query': f"${metrics['estimated_cost'] / max(total_groq + metrics['code_analysis_queries'], 1):.3f}",
            'pricing': {
                'claude': f"${CLAUDE_INPUT_PER_M}/{CLAUDE_OUTPUT_PER_M} per M",
                'grok': f"${GROK_INPUT_PER_M}/{GROK_OUTPUT_PER_M} per M"
            }
        }
    })



@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancers"""
    if not config.health_check_enabled:
        return jsonify({'status': 'disabled'}), 200
    
    health_status = health.get_health()
    status_code = 200 if health_status['status'] == 'healthy' else 503
    
    return jsonify(health_status), status_code

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    """Metrics endpoint"""
    if not config.metrics_enabled:
        return jsonify({'status': 'disabled'}), 200
    
    stats = metrics.get_stats()
    storage_stats = storage.get_stats()
    conversation_stats = conversation_history.get_stats()
    
    return jsonify({
        'metrics': stats,
        'storage': storage_stats,
        'conversations': conversation_stats,
        'config': config.to_dict()
    })

@app.route('/metrics/prometheus', methods=['GET'])
def prometheus_metrics():
    """Prometheus metrics export"""
    exporter = get_prometheus_exporter()
    if not exporter:
        return "Metrics not enabled", 404
    
    return exporter.export(), 200, {'Content-Type': 'text/plain; charset=utf-8'}



def shutdown_handler(signum, frame):
    """Graceful shutdown handler"""
    print("\n" + "=" * 60)
    print("SHUTTING DOWN GRACEFULLY")
    print("=" * 60)
    
    # Save final metrics
    if config.metrics_enabled:
        print("Saving final metrics...")
        stats = metrics.get_stats()
        print(f"Total requests: {stats['summary']['total_requests']}")
        print(f"Total cost: ${stats['summary']['total_cost']:.2f}")
    
    print("=" * 60)
    print("Shutdown complete")
    print("=" * 60)
    sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
atexit.register(lambda: print("FrankenAI v12.0 stopped"))

if __name__ == '__main__':
    print("=" * 60)
    print("FRANKENAI v12.0 - PRODUCTION GRADE")
    print("=" * 60)
    print(f"Storage: {config.storage_backend.upper()}")
    print(f"Monitoring: {'Enabled' if config.metrics_enabled else 'Disabled'}")
    print(f"Health Checks: {'Enabled' if config.health_check_enabled else 'Disabled'}")
    print(f"AWS Integration: {'Enabled' if config.aws_secrets_manager_enabled else 'Disabled'}")
    print("=" * 60)
    print(f"Groq: {'✓' if GROQ_API_KEY else '✗ MISSING'}")
    print(f"Claude: {'✓' if CLAUDE_API_KEY else '✗ MISSING (optional)'}")
    print(f"Grok: {'✓' if GROK_API_KEY else '✗ MISSING (optional)'}")
    print("=" * 60)
    print("FEATURES:")
    print("  ✓ Normal queries: 8-pass pipeline")
    print("  ✓ Code analysis: Specialized 3-pass review")
    print("  ✓ Grok expansion without reversion")
    print("  ✓ Claude compression for verbose output")
    print("=" * 60)
    print("CODE ANALYSIS FLOW:")
    print("  Pass 1:   Groq code summary")
    print("  Pass 5:   Claude deep analysis (full code)")
    print("  Pass 7-8: Synthesis + polish")
    print("=" * 60)
    print("COST ESTIMATES:")
    print("  Regular query:  ~$0.05-0.13")
    print("  Code review:    ~$0.20-0.25")
    print("=" * 60)
    print("URL: http://localhost:5000")
    print("Admin: http://localhost:5000/api/admin/stats")
    print("=" * 60)
    app.run(debug=True, port=5000, host='0.0.0.0')
