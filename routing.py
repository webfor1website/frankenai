"""
FrankenAI v13.0 - Smart Pass Routing
Fixes from v12: Kill wasteful expand→compress loops, adaptive routing
"""

from typing import Dict, Optional, Tuple

def should_revert_grok(
    grok_output: str, 
    original_query: str,
    groq_output: str
) -> Tuple[bool, str]:
    """
    Smart Grok revert decision - only for critical errors
    
    v13 Fix: Only revert on HIGH confidence of hallucination/error
    Not just any "accuracy concern"
    
    Returns:
        (should_revert, reason)
    """
    import re
    
    # Count hallucination markers
    hallucination_markers = [
        (r'according to.*202[6-9]', 'future date citation'),
        (r'recent.*study.*shows', 'vague unsourced claim'),
        (r'scientists.*discovered.*202[6-9]', 'future discovery'),
        (r'research.*proves', 'overconfident claim'),
        (r'breakthrough.*202[6-9]', 'future breakthrough'),
    ]
    
    hallucinations = []
    for pattern, desc in hallucination_markers:
        if re.search(pattern, grok_output.lower()):
            hallucinations.append(desc)
    
    # Only revert if multiple red flags
    if len(hallucinations) >= 3:
        reason = f"High hallucination risk: {', '.join(hallucinations)}"
        print(f"  -> [GROK] {reason}, REVERTING")
        return True, reason
    
    # Check if Grok added technical claims to non-technical query
    from detection import is_technical_query, has_technical_claims
    
    if not is_technical_query(original_query) and has_technical_claims(grok_output):
        # But only if it's SIGNIFICANTLY different from Groq
        groq_has_tech = has_technical_claims(groq_output)
        if not groq_has_tech:
            reason = "Added unsolicited technical claims"
            print(f"  -> [GROK] {reason}, REVERTING")
            return True, reason
    
    # Check for massive unexplained expansion
    groq_words = len(groq_output.split())
    grok_words = len(grok_output.split())
    
    if grok_words > groq_words * 3 and groq_words > 500:
        reason = f"Excessive expansion ({groq_words}→{grok_words} words)"
        print(f"  -> [GROK] {reason}, REVERTING")
        return True, reason
    
    # Otherwise keep it
    print(f"  -> [GROK] Quality acceptable, keeping refinement")
    return False, ""


def smart_pass_routing(
    query: str,
    groq_output: str,
    complexity: Dict,
    target_words: int = 0
) -> Dict:
    """
    Determine optimal pass routing based on output quality
    
    v13 Fix: Single-direction passes, no wasteful loops
    
    Returns routing decisions:
        {
            'skip_claude': bool,
            'skip_grok': bool,
            'skip_frankenai': bool,
            'compress_mode': bool,
            'expand_mode': bool
        }
    """
    output_words = len(groq_output.split())
    
    routing = {
        'skip_claude': False,
        'skip_grok': False,
        'skip_frankenai': False,
        'compress_mode': False,
        'expand_mode': False,
        'reason': ''
    }
    
    # Simple queries: Skip expensive passes
    if complexity['level'] == 'simple':
        routing['skip_grok'] = True
        routing['skip_frankenai'] = True
        routing['reason'] = 'Simple query: Groq + Claude polish only'
        return routing
    
    # Check if Groq output is already good
    if complexity['level'] in ['medium', 'balanced']:
        if 300 <= output_words <= 800:
            routing['skip_grok'] = True
            routing['reason'] = f'Groq output good size ({output_words} words)'
            return routing
    
    # If too short, need expansion
    if output_words < 300 and complexity['needs_expansion']:
        routing['expand_mode'] = True
        routing['skip_grok'] = False
        routing['reason'] = f'Output too short ({output_words} words), expanding'
        return routing
    
    # If too long, need compression (skip expansion)
    if output_words > 1000:
        routing['skip_grok'] = True
        routing['compress_mode'] = True
        routing['reason'] = f'Output too long ({output_words} words), compressing'
        return routing
    
    # Code analysis: Use specialized path
    if complexity['level'] == 'code':
        routing['skip_frankenai'] = True  # Code mode skips synthesis
        routing['reason'] = 'Code analysis: specialized path'
        return routing
    
    # Complex queries: Full pipeline
    if complexity['level'] == 'complex':
        routing['reason'] = 'Complex query: full pipeline'
        return routing
    
    # Default: Medium routing
    routing['reason'] = 'Standard routing'
    return routing


def calculate_estimated_cost(
    passes_used: list,
    groq_tokens: int = 0,
    claude_tokens: int = 0,
    grok_tokens: int = 0
) -> float:
    """
    Calculate estimated API cost for transparency
    
    Rough 2026 pricing:
    - Groq: ~$0.10/$0.30 per 1M tokens
    - Claude: ~$3/$15 per 1M tokens
    - Grok: ~$5/$15 per 1M tokens
    """
    cost = 0.0
    
    # Groq passes (4 passes, ~1500 tokens each input+output)
    if 'groq' in passes_used:
        cost += (groq_tokens / 1_000_000) * 0.20  # Average rate
    
    # Claude passes (~3000 tokens each)
    if 'claude' in passes_used:
        cost += (claude_tokens / 1_000_000) * 9.0  # Average of $3 in + $15 out
    
    # Grok passes (~3000 tokens each)
    if 'grok' in passes_used:
        cost += (grok_tokens / 1_000_000) * 10.0  # Average rate
    
    return round(cost, 4)


def get_optimization_report(
    complexity: Dict,
    routing: Dict,
    passes_executed: list,
    estimated_cost: float
) -> str:
    """
    Generate optimization report for transparency
    
    v13 Feature: Show users what was optimized
    """
    report_lines = [
        f"Complexity: {complexity['level']}",
        f"Passes: {', '.join(passes_executed)}",
        f"Optimizations: {routing['reason']}",
        f"Cost: ~${estimated_cost}"
    ]
    
    skipped = []
    if routing['skip_claude']:
        skipped.append('Claude Pass 5')
    if routing['skip_grok']:
        skipped.append('Grok Pass 6')
    if routing['skip_frankenai']:
        skipped.append('FrankenAI Pass 7')
    
    if skipped:
        report_lines.append(f"Skipped: {', '.join(skipped)}")
    
    return ' | '.join(report_lines)
