"""
PRIORITY 2: LLM-Based Career Fit Evaluation

Use Claude to evaluate career history fit instead of rigid template matching.
This allows nuanced understanding of transferable skills and relevant experience.
"""

import os
import json
from typing import Dict, Tuple, Any


def evaluate_career_with_llm(
    candidate: Dict,
    jd_requirements: Dict[str, Any],
    use_llm: bool = True
) -> Tuple[float, Dict]:
    """
    Evaluate candidate's career fit using LLM.

    Args:
        candidate: Candidate profile dictionary
        jd_requirements: Extracted JD requirements
        use_llm: If False, fall back to rule-based scoring

    Returns:
        Tuple of (career_fit_score, details)
        - career_fit_score: 0-1 score
        - details: Dict with reasoning, strengths, gaps
    """

    if not use_llm:
        # Fallback to original scoring
        from app.services.scoring import score_career
        score = score_career(candidate)
        return score, {"fallback": True}

    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not set, falling back to rule-based career scoring")
            from app.services.scoring import score_career
            return score_career(candidate), {"fallback": True}

        client = anthropic.Anthropic(api_key=api_key)

        # Build career summary
        career_history = candidate.get("career_history", [])
        profile = candidate.get("profile", {})

        career_summary = f"**{profile.get('years_of_experience', 0):.1f} years total experience**\n\n"

        for job in career_history[:5]:  # Top 5 most recent roles
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            duration = job.get("duration_months", 0)
            description = job.get("description", "")
            industry = job.get("industry", "")

            career_summary += f"• **{title}** at {company} ({duration} months)\n"
            if industry:
                career_summary += f"  Industry: {industry}\n"
            if description:
                career_summary += f"  {description[:200]}...\n" if len(description) > 200 else f"  {description}\n"
            career_summary += "\n"

        # Build requirement summary
        req_summary = f"""**Role Requirements:**
- Level: {jd_requirements.get('role_level', 'Senior')}
- Experience: {jd_requirements.get('experience_range', {}).get('min', 5)}-{jd_requirements.get('experience_range', {}).get('max', 9)} years
- Key skills: {', '.join(jd_requirements.get('must_have_skills', [])[:8])}
- Responsibilities: {'; '.join(jd_requirements.get('key_responsibilities', [])[:3])}
"""

        prompt = f"""You are an expert technical recruiter evaluating a candidate for a Senior AI Engineer position focused on vector search, embeddings, RAG, and ranking systems.

{req_summary}

**Candidate Career History:**
{career_summary}

Evaluate this candidate's career fit based on:

1. **Relevant technical experience** (40%): Direct experience with vector search, embeddings, RAG, ranking, ML systems
2. **Production deployment** (25%): Evidence of shipping ML/AI systems to production, handling scale
3. **Career trajectory** (20%): Growth, increasing responsibility, seniority
4. **Industry alignment** (15%): Product companies vs consulting, AI/ML-focused roles

Return ONLY a JSON object:
{{
  "career_fit_score": <float 0.0-1.0>,
  "key_strengths": [<2-3 specific strengths>],
  "gaps": [<1-2 gaps or concerns, if any>],
  "reasoning": "<1-2 sentence summary>",
  "production_experience": <boolean>,
  "relevant_companies": <int 0-5>
}}

Be realistic and calibrated. A perfect 1.0 score should be rare (exceptional fit). Most strong candidates should score 0.6-0.85."""

        message = client.messages.create(
            model="claude-3-5-haiku-20241022",  # Fast, cost-effective
            max_tokens=800,
            temperature=0.3,  # Slight creativity for better evaluation
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        career_score = float(result.get("career_fit_score", 0.5))
        career_score = max(0.0, min(career_score, 1.0))  # Clamp to [0, 1]

        details = {
            "llm_evaluation": True,
            "key_strengths": result.get("key_strengths", []),
            "gaps": result.get("gaps", []),
            "reasoning": result.get("reasoning", ""),
            "production_experience": result.get("production_experience", False),
            "relevant_companies": result.get("relevant_companies", 0)
        }

        return career_score, details

    except Exception as e:
        print(f"Warning: LLM career evaluation failed ({e}), using fallback")
        from app.services.scoring import score_career
        return score_career(candidate), {"fallback": True, "error": str(e)}


def batch_evaluate_careers_with_llm(
    candidates: list,
    jd_requirements: Dict[str, Any],
    use_llm: bool = True,
    max_batch: int = 500
) -> Dict[str, Tuple[float, Dict]]:
    """
    Batch evaluate multiple candidates' careers.

    Args:
        candidates: List of candidate dictionaries
        jd_requirements: JD requirements
        use_llm: Whether to use LLM
        max_batch: Maximum candidates to evaluate with LLM

    Returns:
        Dict mapping candidate_id -> (score, details)
    """
    results = {}

    # Only use LLM for top candidates (already pre-filtered)
    for i, candidate in enumerate(candidates[:max_batch]):
        cid = candidate.get("candidate_id", f"unknown_{i}")

        if i > 0 and i % 50 == 0:
            print(f"  LLM career eval progress: {i}/{min(len(candidates), max_batch)}")

        score, details = evaluate_career_with_llm(candidate, jd_requirements, use_llm)
        results[cid] = (score, details)

    # For remaining candidates beyond max_batch, use fallback
    if len(candidates) > max_batch:
        print(f"  Using fallback for {len(candidates) - max_batch} candidates beyond batch limit")
        from app.services.scoring import score_career

        for candidate in candidates[max_batch:]:
            cid = candidate.get("candidate_id", "unknown")
            score = score_career(candidate)
            results[cid] = (score, {"fallback": True, "reason": "beyond_batch_limit"})

    return results
