"""
FrankenAI v13.0 - Strict Word Count Enforcement
Fixes from v12: Iterative enforcement with hard limits
"""

import re
from typing import Optional

def count_words(text: str) -> int:
    """Accurate word count"""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Count words
    return len(text.split())


def enforce_word_count_strict(
    text: str, 
    target: int, 
    tolerance: int = 5,
    max_attempts: int = 3,
    groq_client=None
) -> str:
    """
    Strictly enforce word count with iterative adjustments
    
    v13 Fix: Multiple attempts with hard cut/pad as fallback
    
    Args:
        text: Input text
        target: Target word count
        tolerance: Acceptable deviation (default 5 words)
        max_attempts: Max adjustment attempts before hard cut/pad
        groq_client: Groq client for AI adjustments
    
    Returns:
        Text adjusted to target ± tolerance words
    """
    current = count_words(text)
    
    # Already within tolerance
    if abs(current - target) <= tolerance:
        print(f"  -> Word count OK: {current} (target {target}±{tolerance})")
        return text
    
    print(f"  -> Word count OFF: {current} vs target {target}, adjusting...")
    
    # Attempt AI adjustments
    for attempt in range(max_attempts):
        diff = target - current
        
        if abs(diff) <= tolerance:
            print(f"  -> Adjusted to {current} words (attempt {attempt + 1})")
            return text
        
        # Calculate adjustment prompt
        if diff > 0:
            # Need more words
            adjustment = diff
            prompt = f"""Expand this text to exactly {target} words by adding relevant detail and elaboration. Current: {current} words, need {adjustment} more.

Text: {text}

Output ONLY the expanded text, no preamble."""
        else:
            # Need fewer words
            adjustment = abs(diff)
            prompt = f"""Compress this text to exactly {target} words by removing redundancy and being more concise. Current: {current} words, need to cut {adjustment} words.

Text: {text}

Output ONLY the compressed text, no preamble."""
        
        # Try AI adjustment
        if groq_client:
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                text = response.choices[0].message.content.strip()
                current = count_words(text)
                print(f"  -> AI adjustment {attempt + 1}: {current} words")
            except Exception as e:
                print(f"  -> AI adjustment failed: {e}")
                break
        else:
            break
    
    # If still wrong after attempts, HARD CUT or PAD
    final_count = count_words(text)
    
    if abs(final_count - target) > tolerance:
        print(f"  -> AI adjustments insufficient, applying HARD adjustment")
        text = hard_adjust_word_count(text, target)
        final_count = count_words(text)
        print(f"  -> Final count: {final_count} words")
    
    return text


def hard_adjust_word_count(text: str, target: int) -> str:
    """
    Hard cut or pad to exact word count
    Last resort when AI adjustments fail
    """
    words = text.split()
    current = len(words)
    
    if current > target:
        # HARD CUT: Keep first target words, add period
        result = ' '.join(words[:target])
        # Try to end on a sentence
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    elif current < target:
        # HARD PAD: Add philosophical filler
        deficit = target - current
        
        padding_phrases = [
            "This further demonstrates the complexity and nuance of the matter.",
            "Additional considerations include the broader implications and context.",
            "The philosophical dimensions of this question merit further exploration.",
            "These factors collectively contribute to a more complete understanding.",
            "Further analysis reveals deeper layers of meaning and significance.",
        ]
        
        # Add padding phrases until we reach target
        padded = text
        phrase_idx = 0
        words_added = 0
        
        while words_added < deficit:
            phrase = padding_phrases[phrase_idx % len(padding_phrases)]
            phrase_words = phrase.split()
            
            if words_added + len(phrase_words) <= deficit:
                padded += ' ' + phrase
                words_added += len(phrase_words)
                phrase_idx += 1
            else:
                # Add partial phrase
                remaining = deficit - words_added
                partial = ' '.join(phrase_words[:remaining])
                padded += ' ' + partial + '.'
                words_added = deficit
        
        return padded
    
    return text


def check_word_count_compliance(text: str, target: int, tolerance: int = 5) -> bool:
    """Check if text meets word count requirements"""
    if target == 0:
        return True
    
    current = count_words(text)
    compliant = abs(current - target) <= tolerance
    
    if not compliant:
        print(f"  -> ⚠️ Word count non-compliant: {current} vs {target}±{tolerance}")
    
    return compliant
