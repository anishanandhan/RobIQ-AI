"""
PRIORITY 3: Hybrid Ranking Pipeline with Reranking

Two-stage ranking system:
1. Fast rule-based scoring to filter 100k → 500 candidates
2. LLM-based reranking for top 500 → final 100

This balances speed, cost, and quality.
"""

from typing import List, Dict, Tuple, Any, Optional
import time


def hybrid_ranking_pipeline(
    candidates: List[Dict],
    jd_requirements: Optional[Dict[str, Any]] = None,
    top_n: int = 100,
    rerank_pool: int = 500,
    use_semantic: bool = True,
    use_llm_career: bool = True,
    use_llm_rerank: bool = True,
    llm_provider: str = "anthropic"
) -> List[Tuple[float, Dict, Dict]]:
    """
    Hybrid two-stage ranking pipeline.

    Stage 1: Fast rule-based scoring (all candidates)
    Stage 2: Enhanced scoring with semantic + LLM (top 500)

    Args:
        candidates: List of candidate dictionaries
        jd_requirements: Extracted JD requirements (if None, will use defaults)
        top_n: Final number of candidates to return
        rerank_pool: Number of candidates to re-rank with enhanced methods
        use_semantic: Use semantic skill matching in stage 2
        use_llm_career: Use LLM career evaluation in stage 2
        use_llm_rerank: Use LLM for final reranking

    Returns:
        List of (score, candidate, breakdown) tuples, sorted by score descending
    """
    from app.services.scoring import score_candidate
    from app.services.jd_parser import get_jd_requirements
    from app.services.semantic_matcher import get_semantic_matcher
    from app.services.llm_evaluator import evaluate_career_with_llm

    print(f"\n{'='*80}")
    print("HYBRID RANKING PIPELINE")
    print(f"{'='*80}")
    print(f"Total candidates: {len(candidates):,}")
    print(f"Rerank pool size: {rerank_pool}")
    print(f"Final output: Top {top_n}")
    print(f"Semantic matching: {use_semantic}")
    print(f"LLM career eval: {use_llm_career}")
    print(f"LLM reranking: {use_llm_rerank}")
    print(f"{'='*80}\n")

    # Load JD requirements if not provided
    if jd_requirements is None:
        jd_requirements = get_jd_requirements()

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: Fast Rule-Based Scoring (100k → 500)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"[Stage 1] Fast rule-based scoring on {len(candidates):,} candidates...")
    stage1_start = time.time()

    stage1_scored = []
    stage1_errors = 0

    for i, candidate in enumerate(candidates):
        try:
            score, breakdown = score_candidate(candidate)
            stage1_scored.append((score, candidate, breakdown))

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,} candidates...")

        except Exception as e:
            stage1_errors += 1
            if stage1_errors < 5:  # Only print first few errors
                print(f"  Warning: Scoring error for candidate {i}: {e}")

    # Sort by score descending, candidate_id ascending (tie-breaker)
    stage1_scored.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))

    stage1_time = time.time() - stage1_start
    print(f"✓ Stage 1 complete in {stage1_time:.1f}s")
    print(f"  Scored: {len(stage1_scored):,} | Errors: {stage1_errors}")

    if len(stage1_scored) > 0:
        print(f"  Score range: {stage1_scored[0][0]:.4f} → {stage1_scored[-1][0]:.4f}")

    # Take top candidates for reranking
    rerank_candidates = stage1_scored[:rerank_pool]
    print(f"  Top {len(rerank_candidates)} candidates selected for reranking")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: Enhanced Scoring (500 → 100)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[Stage 2] Enhanced scoring with semantic + LLM on top {len(rerank_candidates)}...")
    stage2_start = time.time()

    # Initialize semantic matcher if needed
    semantic_matcher = None
    if use_semantic:
        print("  Loading semantic skill matcher...")
        semantic_matcher = get_semantic_matcher()

    reranked = []

    for i, (stage1_score, candidate, stage1_breakdown) in enumerate(rerank_candidates):
        try:
            enhanced_score = stage1_score
            enhanced_breakdown = stage1_breakdown.copy()

            # Semantic skill matching
            if use_semantic and semantic_matcher:
                semantic_score, semantic_details = semantic_matcher.score_candidate_skills_semantic(
                    candidate,
                    must_have_skills=jd_requirements.get("must_have_skills", []),
                    nice_to_have_skills=jd_requirements.get("nice_to_have_skills", [])
                )
                enhanced_breakdown["semantic_skill"] = round(semantic_score, 3)
                enhanced_breakdown["semantic_details"] = semantic_details

                # Blend: 50% original skill score + 50% semantic
                original_skill = stage1_breakdown.get("skill", 0.5)
                blended_skill = 0.5 * original_skill + 0.5 * semantic_score
                enhanced_breakdown["skill"] = round(blended_skill, 3)

            # LLM career evaluation
            if use_llm_career:
                # Choose LLM provider
                if llm_provider == "google" or llm_provider == "gemini":
                    from app.services.gemini_evaluator import evaluate_career_with_gemini
                    llm_career_score, llm_career_details = evaluate_career_with_gemini(
                        candidate,
                        jd_requirements,
                        use_llm=True
                    )
                else:  # default to anthropic
                    llm_career_score, llm_career_details = evaluate_career_with_llm(
                        candidate,
                        jd_requirements,
                        use_llm=True
                    )

                if not llm_career_details.get("fallback"):
                    enhanced_breakdown["llm_career"] = round(llm_career_score, 3)
                    enhanced_breakdown["llm_career_details"] = llm_career_details

                    # Blend: 60% LLM + 40% original
                    original_career = stage1_breakdown.get("career", 0.5)
                    blended_career = 0.6 * llm_career_score + 0.4 * original_career
                    enhanced_breakdown["career"] = round(blended_career, 3)

            # Recompute final score with enhanced components
            skill = enhanced_breakdown.get("skill", stage1_breakdown.get("skill", 0.5))
            career = enhanced_breakdown.get("career", stage1_breakdown.get("career", 0.5))
            behavioral = enhanced_breakdown.get("behavioral", stage1_breakdown.get("behavioral", 0.5))
            dq_mult = enhanced_breakdown.get("dq_mult", 1.0)

            # Weighted composite (same weights as original)
            enhanced_score = (0.45 * skill + 0.35 * career + 0.20 * behavioral) * dq_mult

            reranked.append((enhanced_score, candidate, enhanced_breakdown))

            if (i + 1) % 100 == 0:
                print(f"  Enhanced scoring progress: {i+1}/{len(rerank_candidates)}")

        except Exception as e:
            print(f"  Warning: Enhanced scoring error for candidate {i}: {e}")
            # Fall back to stage 1 score
            reranked.append((stage1_score, candidate, stage1_breakdown))

    # Sort reranked candidates
    reranked.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))

    stage2_time = time.time() - stage2_start
    print(f"✓ Stage 2 complete in {stage2_time:.1f}s")

    if len(reranked) > 0:
        print(f"  Enhanced score range: {reranked[0][0]:.4f} → {reranked[-1][0]:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL: Return Top N
    # ──────────────────────────────────────────────────────────────────────────
    final_results = reranked[:top_n]

    total_time = stage1_time + stage2_time
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final output: {len(final_results)} candidates")
    if len(final_results) > 0:
        print(f"Score range: {final_results[0][0]:.4f} → {final_results[-1][0]:.4f}")
    print(f"{'='*80}\n")

    return final_results
