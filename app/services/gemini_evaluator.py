"""
LLM-Based Career Fit Evaluation using Google Gemini

Alternative to Anthropic Claude for career assessment.
"""

import os
import json
from typing import Dict, Tuple, Any


def evaluate_career_with_gemini(
    candidate: Dict,
    jd_requirements: Dict[str, Any],
    use_llm: bool = True
) -> Tuple[float, Dict]:
    """
    Evaluate candidate's career fit using Google Gemini.

    Args:
        candidate: Candidate profile dictionary
        jd_requirements: Extracted JD requirements
        use_llm: If False, fall back to rule-based scoring

    Returns:
        Tuple of (career_fit_score, details)
    """

    if not use_llm:
        from app.services.scoring import score_career
        score = score_career(candidate)
        return score, {"fallback": True}

    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY not set, falling back to rule-based career scoring")
            from app.services.scoring import score_career
            return score_career(candidate), {"fallback": True}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Build career summary
        career_history = candidate.get("career_history", [])
        profile = candidate.get("profile", {})

        career_summary = f"**{profile.get('years_of_experience', 0):.1f} years total experience**\n\n"

        for job in career_history[:5]:
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

Return ONLY a JSON object (no markdown, no extra text):
{{
  "career_fit_score": <float 0.0-1.0>,
  "key_strengths": [<2-3 specific strengths>],
  "gaps": [<1-2 gaps or concerns, if any>],
  "reasoning": "<1-2 sentence summary>",
  "production_experience": <boolean>,
  "relevant_companies": <int 0-5>
}}

Be realistic and calibrated. A perfect 1.0 score should be rare (exceptional fit). Most strong candidates should score 0.6-0.85."""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=800,
            )
        )

        response_text = response.text.strip()

        # Clean up response - remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        career_score = float(result.get("career_fit_score", 0.5))
        career_score = max(0.0, min(career_score, 1.0))

        details = {
            "llm_evaluation": True,
            "llm_provider": "google_gemini",
            "key_strengths": result.get("key_strengths", []),
            "gaps": result.get("gaps", []),
            "reasoning": result.get("reasoning", ""),
            "production_experience": result.get("production_experience", False),
            "relevant_companies": result.get("relevant_companies", 0)
        }

        return career_score, details

    except Exception as e:
        print(f"Warning: Gemini career evaluation failed ({e}), using fallback")
        from app.services.scoring import score_career
        return score_career(candidate), {"fallback": True, "error": str(e)}


def batch_evaluate_careers_with_gemini(
    candidates: list,
    jd_requirements: Dict[str, Any],
    use_llm: bool = True,
    max_batch: int = 500
) -> Dict[str, Tuple[float, Dict]]:
    """
    Batch evaluate multiple candidates' careers with Gemini.

    Args:
        candidates: List of candidate dictionaries
        jd_requirements: JD requirements
        use_llm: Whether to use LLM
        max_batch: Maximum candidates to evaluate with LLM

    Returns:
        Dict mapping candidate_id -> (score, details)
    """
    results = {}

    for i, candidate in enumerate(candidates[:max_batch]):
        cid = candidate.get("candidate_id", f"unknown_{i}")

        if i > 0 and i % 50 == 0:
            print(f"  Gemini career eval progress: {i}/{min(len(candidates), max_batch)}")

        score, details = evaluate_career_with_gemini(candidate, jd_requirements, use_llm)
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
