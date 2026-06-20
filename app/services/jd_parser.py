"""
PRIORITY 4: Extract Job Description Requirements

Dynamically parse job descriptions to extract structured requirements
instead of relying on hardcoded skill lists.
"""

import os
import json
from typing import Dict, Any, Optional
from docx import Document


def extract_jd_text(jd_path: str) -> str:
    """Extract text from a .docx job description file."""
    try:
        doc = Document(jd_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Warning: Could not parse JD file {jd_path}: {e}")
        return ""


def extract_jd_requirements_with_llm(jd_text: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Extract structured requirements from job description text.

    Args:
        jd_text: The full job description text
        use_llm: If True, use Anthropic Claude; if False, return defaults

    Returns:
        Dictionary with:
        - must_have_skills: List of critical technical skills
        - nice_to_have_skills: List of preferred skills
        - experience_range: {"min": years, "max": years}
        - key_responsibilities: List of main responsibilities
        - location_requirements: Location info
        - role_level: seniority level
    """
    if not use_llm or not jd_text:
        # Return sensible defaults based on the challenge
        return {
            "must_have_skills": [
                "embedding", "vector search", "faiss", "qdrant", "pinecone",
                "rag", "ranking", "retrieval", "llm", "python"
            ],
            "nice_to_have_skills": [
                "lora", "fine-tuning", "recommendation", "distributed systems"
            ],
            "experience_range": {"min": 5, "max": 9},
            "key_responsibilities": [
                "Build vector search and ranking systems",
                "Deploy ML models to production",
                "Work with embeddings and RAG pipelines"
            ],
            "location_requirements": "India (Noida/Pune - hybrid)",
            "role_level": "Senior"
        }

    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not set, using defaults")
            return extract_jd_requirements_with_llm(jd_text, use_llm=False)

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Extract the key requirements from this job description and return ONLY a valid JSON object.

<job_description>
{jd_text}
</job_description>

Analyze the JD and return a JSON object with these fields:
- must_have_skills: Array of 8-12 critical technical skills (lowercase, specific tech/tools)
- nice_to_have_skills: Array of 5-8 preferred skills
- experience_range: Object with min and max years
- key_responsibilities: Array of 3-5 main responsibilities
- location_requirements: String describing location
- role_level: String (Junior/Mid/Senior/Staff/Lead)

Return ONLY the JSON object, no other text."""

        message = client.messages.create(
            model="claude-3-5-haiku-20241022",  # Fast and cost-effective
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        print("✓ Successfully extracted JD requirements via LLM")
        return result

    except Exception as e:
        print(f"Warning: LLM extraction failed ({e}), using defaults")
        return extract_jd_requirements_with_llm(jd_text, use_llm=False)


def load_jd_requirements(jd_path: Optional[str] = None, use_llm: bool = True) -> Dict[str, Any]:
    """
    Main entry point: Load and parse job description.

    Args:
        jd_path: Path to .docx job description file
        use_llm: Whether to use LLM for extraction (default True)

    Returns:
        Structured requirements dictionary
    """
    if jd_path and os.path.exists(jd_path):
        jd_text = extract_jd_text(jd_path)
        if jd_text:
            return extract_jd_requirements_with_llm(jd_text, use_llm=use_llm)

    # Fallback to defaults
    print("Warning: Using default JD requirements (no file provided or parse failed)")
    return extract_jd_requirements_with_llm("", use_llm=False)


# Cache for JD requirements to avoid re-parsing
_jd_cache: Optional[Dict[str, Any]] = None


def get_jd_requirements(jd_path: Optional[str] = None, use_llm: bool = True, force_reload: bool = False) -> Dict[str, Any]:
    """
    Get JD requirements with caching.

    Args:
        jd_path: Path to JD file
        use_llm: Whether to use LLM
        force_reload: Force re-parsing even if cached

    Returns:
        Cached or freshly parsed JD requirements
    """
    global _jd_cache

    if _jd_cache is not None and not force_reload:
        return _jd_cache

    _jd_cache = load_jd_requirements(jd_path, use_llm)
    return _jd_cache
