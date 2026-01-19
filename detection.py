"""
FrankenAI v13.0 - Enhanced Query Detection
Fixes from v12: Better code detection, smarter complexity assessment
"""

import re
from typing import Dict, Tuple

def detect_code_analysis(query: str) -> bool:
    """
    Improved code detection with multiple signals
    
    v13 Fix: Multi-signal approach instead of keyword-only
    Catches truncated code, React hooks, and review requests
    """
    code_signals = 0
    query_lower = query.lower()
    
    # Signal 1: Code blocks (weight: 2)
    if '```' in query or 'import ' in query or 'def ' in query or 'function ' in query:
        code_signals += 2
    
    # Signal 2: Programming keywords (weight: 1 each, max 5)
    code_keywords = [
        'function', 'class', 'const', 'let', 'var', 'return',
        'useeffect', 'usestate', 'async', 'await', 'export',
        'import', 'require', 'module', 'package'
    ]
    keyword_count = min(5, sum(1 for kw in code_keywords if kw in query_lower))
    code_signals += keyword_count
    
    # Signal 3: Code review language (weight: 2)
    review_terms = ['review', 'analyze', 'refactor', 'improve', 'debug', 
                    'fix', 'optimize', 'check']
    if any(term in query_lower for term in review_terms):
        code_signals += 2
    
    # Signal 4: Code structure patterns (weight: 3)
    structure_patterns = [
        r'\bdef\s+\w+\s*\(',           # Python function
        r'\bfunction\s+\w+\s*\(',      # JS function
        r'\bclass\s+\w+',              # Class definition
        r'\=\>\s*{',                   # Arrow function
        r'useState\(',                 # React hooks
        r'useEffect\(',                # React hooks
    ]
    if any(re.search(pattern, query) for pattern in structure_patterns):
        code_signals += 3
    
    # Signal 5: Multiple code lines (weight: 2)
    lines = query.split('\n')
    code_like_lines = sum(1 for line in lines if ':' in line or '{' in line or ';' in line)
    if code_like_lines >= 5:
        code_signals += 2
    
    # Threshold: 4+ signals = code mode
    is_code = code_signals >= 4
    
    if is_code:
        print(f"  -> [DETECTION] Code analysis mode (signals: {code_signals})")
    
    return is_code


def assess_query_complexity(query: str, query_type: str) -> Dict:
    """
    Assess query complexity and determine optimal pass count
    
    v13 Fix: Adaptive pass allocation instead of always 8
    """
    complexity = {
        'level': 'medium',
        'pass_count': 4,
        'needs_expansion': True,
        'needs_compression': True,
        'early_exit_ok': False
    }
    
    query_lower = query.lower()
    word_count = len(query.split())
    
    # Trivial queries (instant response)
    trivial_patterns = [
        r'^what\'?s?\s+\d+\s*[\+\-\*\/]\s*\d+',  # Math: "what's 2+2"
        r'^(hi|hello|hey|yo)[\s\?!\.]*$',         # Greetings
    ]
    if any(re.match(pattern, query_lower) for pattern in trivial_patterns):
        complexity.update({
            'level': 'trivial',
            'pass_count': 0,
            'needs_expansion': False,
            'needs_compression': False,
            'early_exit_ok': True
        })
        return complexity
    
    # Simple queries (2-3 passes)
    if query_type == 'simple' or word_count < 20:
        complexity.update({
            'level': 'simple',
            'pass_count': 3,
            'needs_expansion': False,
            'needs_compression': True,
            'early_exit_ok': True
        })
        return complexity
    
    # Code analysis (fixed 5 passes)
    if detect_code_analysis(query):
        complexity.update({
            'level': 'code',
            'pass_count': 5,
            'needs_expansion': True,
            'needs_compression': False,
            'early_exit_ok': False
        })
        return complexity
    
    # Medium complexity (4-5 passes)
    if query_type in ['balanced', 'creative'] and word_count < 50:
        complexity.update({
            'level': 'medium',
            'pass_count': 4,
            'needs_expansion': True,
            'needs_compression': True,
            'early_exit_ok': True
        })
        return complexity
    
    # Complex/analytical (6-8 passes)
    complexity_indicators = [
        'prove', 'disprove', 'exhaustive', 'comprehensive',
        'rigorous', 'formal', 'philosophical', 'explain from first principles'
    ]
    is_complex = any(ind in query_lower for ind in complexity_indicators)
    
    if is_complex or query_type in ['technical', 'philosophical', 'analytical']:
        pass_count = 8 if 'exhaustive' in query_lower or 'comprehensive' in query_lower else 6
        complexity.update({
            'level': 'complex',
            'pass_count': pass_count,
            'needs_expansion': True,
            'needs_compression': True,
            'early_exit_ok': False
        })
        return complexity
    
    return complexity


def extract_word_count_target(query: str) -> int:
    """
    Extract target word count from query
    
    Patterns: "exactly 420 words", "must be 666 words", "420-word"
    """
    patterns = [
        r'exactly\s+(\d+)\s+words?',
        r'must\s+be\s+(\d+)\s+words?',
        r'(\d+)-word',
        r'(\d+)\s+words?\s+(no\s+more|no\s+less|exactly)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            target = int(match.group(1))
            print(f"  -> [DETECTION] Word count target: {target}")
            return target
    
    return 0


def should_skip_pass(pass_name: str, complexity: Dict, current_output: str) -> bool:
    """
    Determine if a pass should be skipped based on complexity and current output
    
    v13 Fix: Smart pass skipping to save cost
    """
    output_words = len(current_output.split())
    
    # Never skip on complex queries
    if complexity['level'] == 'complex':
        return False
    
    # Skip expansion if output is already good size
    if pass_name == 'grok_expansion' and not complexity['needs_expansion']:
        print(f"  -> Skipping Grok expansion (not needed)")
        return True
    
    if pass_name == 'grok_expansion' and 300 <= output_words <= 800:
        print(f"  -> Skipping Grok expansion (output size good: {output_words} words)")
        return True
    
    # Skip compression if output is concise
    if pass_name == 'claude_compression' and output_words < 500:
        print(f"  -> Skipping Claude compression (already concise: {output_words} words)")
        return True
    
    # Allow early exit for simple queries with high quality
    if complexity['early_exit_ok'] and output_words >= 200:
        return pass_name in ['frankenai_synthesis', 'claude_finisher']
    
    return False


def is_technical_query(query: str) -> bool:
    """Check if query requests technical/scientific analysis"""
    technical_terms = [
        'algorithm', 'complexity', 'theorem', 'proof', 'formal',
        'quantum', 'physics', 'mathematics', 'computational'
    ]
    return any(term in query.lower() for term in technical_terms)


def has_technical_claims(text: str) -> bool:
    """Check if text contains unsolicited technical claims"""
    claim_patterns = [
        r'according to.*study',
        r'research shows',
        r'scientists.*discovered',
        r'proven that',
        r'\d{4}\s+paper'
    ]
    return sum(1 for pattern in claim_patterns if re.search(pattern, text.lower())) >= 2
