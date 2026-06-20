"""
PRIORITY 1: Semantic Skill Matching with Sentence Transformers

Use embedding-based semantic similarity instead of keyword matching
to better understand skill relevance.
"""

from typing import List, Dict, Tuple, Optional, Any


class SemanticSkillMatcher:
    """
    Semantic skill matcher using sentence-transformers.
    Caches embeddings for efficiency.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic matcher.

        Args:
            model_name: Sentence transformer model name
                - 'all-MiniLM-L6-v2': Fast, lightweight (80MB)
                - 'all-mpnet-base-v2': Better quality, slower (420MB)
        """
        from sentence_transformers import SentenceTransformer

        print(f"Loading semantic skill matcher: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.skill_embeddings_cache: Dict[str, np.ndarray] = {}
        print("✓ Semantic matcher ready")

    def encode_skills(self, skills: List[str]) -> np.ndarray:
        """
        Encode a list of skill strings into embeddings.

        Args:
            skills: List of skill names/descriptions

        Returns:
            numpy array of shape (len(skills), embedding_dim)
        """
        return self.model.encode(skills, convert_to_numpy=True, show_progress_bar=False)

    def compute_skill_similarity(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        threshold: float = 0.65
    ) -> Tuple[float, List[Tuple[str, str, float]]]:
        """
        Compute semantic similarity between candidate and required skills.

        Args:
            candidate_skills: List of skills from candidate profile
            required_skills: List of required skills from JD
            threshold: Similarity threshold (0.65 = strong semantic match)

        Returns:
            Tuple of (overall_score, matches)
            - overall_score: 0-1 score based on coverage
            - matches: List of (required_skill, matched_candidate_skill, similarity)
        """
        if not candidate_skills or not required_skills:
            return 0.0, []

        # Encode both skill sets
        candidate_embs = self.encode_skills(candidate_skills)
        required_embs = self.encode_skills(required_skills)

        # Compute similarity matrix: (n_required, n_candidate)
        similarity_matrix = cosine_similarity(required_embs, candidate_embs)

        matches = []
        matched_required = set()

        # For each required skill, find best candidate match
        for i, req_skill in enumerate(required_skills):
            similarities = similarity_matrix[i]
            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]

            if best_sim >= threshold:
                matches.append((req_skill, candidate_skills[best_idx], float(best_sim)))
                matched_required.add(i)

        # Overall score: fraction of required skills matched
        coverage_score = len(matched_required) / len(required_skills) if required_skills else 0.0

        # Bonus for strong matches (sim > 0.8)
        strong_matches = sum(1 for _, _, sim in matches if sim > 0.8)
        quality_bonus = min(strong_matches * 0.05, 0.2)  # Up to +0.2 bonus

        overall_score = min(coverage_score + quality_bonus, 1.0)

        return overall_score, matches

    def score_candidate_skills_semantic(
        self,
        candidate: Dict,
        must_have_skills: List[str],
        nice_to_have_skills: List[str],
        must_threshold: float = 0.65,
        nice_threshold: float = 0.60
    ) -> Tuple[float, Dict]:
        """
        Score candidate skills using semantic matching.

        Args:
            candidate: Candidate profile dictionary
            must_have_skills: Required skills from JD
            nice_to_have_skills: Preferred skills from JD
            must_threshold: Similarity threshold for must-have (higher bar)
            nice_threshold: Similarity threshold for nice-to-have

        Returns:
            Tuple of (score, breakdown)
            - score: 0-1 overall skill score
            - breakdown: Dict with details
        """
        from app.utils.helpers import skill_names, full_text

        # Extract candidate skills
        candidate_skill_list = skill_names(candidate)

        # Also extract skill mentions from text (career descriptions, summary)
        full = full_text(candidate)
        # Simple extraction: split on common delimiters and take skill-like terms
        text_skills = [
            word.strip() for word in full.split()
            if len(word.strip()) > 3 and word.strip().isalpha()
        ]
        all_candidate_skills = list(set(candidate_skill_list + text_skills[:50]))  # Limit to avoid noise

        # Semantic matching for must-have skills
        must_score, must_matches = self.compute_skill_similarity(
            all_candidate_skills,
            must_have_skills,
            threshold=must_threshold
        )

        # Semantic matching for nice-to-have skills
        nice_score, nice_matches = self.compute_skill_similarity(
            all_candidate_skills,
            nice_to_have_skills,
            threshold=nice_threshold
        )

        # Weighted composite
        final_score = 0.75 * must_score + 0.25 * nice_score

        breakdown = {
            "must_have_score": round(must_score, 3),
            "nice_to_have_score": round(nice_score, 3),
            "must_have_matches": len(must_matches),
            "nice_to_have_matches": len(nice_matches),
            "top_matches": must_matches[:5]  # Top 5 matches for debugging
        }

        return final_score, breakdown


# Global singleton instance
_semantic_matcher: Optional[SemanticSkillMatcher] = None


def get_semantic_matcher(model_name: str = 'all-MiniLM-L6-v2') -> SemanticSkillMatcher:
    """
    Get or create global semantic matcher instance.

    Args:
        model_name: Model to use

    Returns:
        SemanticSkillMatcher instance
    """
    global _semantic_matcher

    if _semantic_matcher is None:
        _semantic_matcher = SemanticSkillMatcher(model_name)

    return _semantic_matcher
