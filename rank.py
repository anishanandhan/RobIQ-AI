#!/usr/bin/env python3
"""
RedrObIQ Ranker — Intelligent Candidate Ranking for Senior AI Engineer (Founding Team)
by Anish (VIT Chennai / CyStar IIT Madras)

Enhanced with:
- Priority 1: Semantic skill matching (sentence-transformers)
- Priority 2: LLM-based career evaluation (Claude)
- Priority 3: Hybrid ranking pipeline (fast filter + LLM rerank)
- Priority 4: Dynamic job description parsing
- Priority 5: Skill synonym matching

This script acts as the entry point coordinator for parsing arguments,
reading input profiles, scoring candidates via the app modular engine,
and writing the sorted, formatted output shortlist.
"""

import json
import csv
import sys
import argparse
import os
from app.services.scoring import score_candidate
from app.services.reasoning import generate_reasoning

def main():
    parser = argparse.ArgumentParser(
        description="RedrObIQ Candidate Ranker CLI - Enhanced with Semantic + LLM Ranking"
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to return")

    # Enhanced features
    parser.add_argument("--jd", help="Path to job description .docx file (optional)")
    parser.add_argument(
        "--mode",
        choices=["fast", "hybrid", "enhanced"],
        default="fast",
        help="Ranking mode: 'fast' (original), 'hybrid' (semantic+LLM on top 500), 'enhanced' (full LLM)"
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=500,
        help="Number of candidates to rerank with LLM in hybrid mode"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic skill matching"
    )
    parser.add_argument(
        "--no-llm-career",
        action="store_true",
        help="Disable LLM career evaluation"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "google", "gemini"],
        default="anthropic",
        help="LLM provider for career evaluation (anthropic, google/gemini)"
    )

    args = parser.parse_args()

    # Display mode info
    print(f"\n{'='*80}")
    print("RedrObIQ Enhanced Candidate Ranker")
    print(f"{'='*80}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Candidates: {args.candidates}")
    print(f"Output: {args.out}")
    print(f"Top N: {args.top}")
    if args.jd:
        print(f"Job Description: {args.jd}")
    if args.mode == "hybrid":
        print(f"Rerank pool: {args.rerank_pool}")
        print(f"Semantic matching: {not args.no_semantic}")
        print(f"LLM career eval: {not args.no_llm_career}")
    print(f"{'='*80}\n")

    # Check for ANTHROPIC_API_KEY if using LLM features
    if args.mode in ["hybrid", "enhanced"] and not args.no_llm_career:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  Warning: ANTHROPIC_API_KEY not set. LLM features will be disabled.")
            print("   Set with: export ANTHROPIC_API_KEY='your-key-here'\n")

    # Load candidates
    print(f"Loading candidates from {args.candidates} ...")

    candidates = []
    errors = 0

    try:
        with open(args.candidates, "r", encoding="utf-8") as f:
            # Peek at the first character to determine format
            first_char = ""
            for char in f.read(100):
                if not char.isspace():
                    first_char = char
                    break
            f.seek(0)

            if first_char == "[":
                # Parse as a single JSON array
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        candidates = data
                    else:
                        candidates = [data]
                except Exception as e:
                    print(f"Error parsing JSON array: {e}")
                    sys.exit(1)
            else:
                # Parse line-by-line (JSONL)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        candidate = json.loads(line)
                        candidates.append(candidate)

                        if len(candidates) % 10000 == 0:
                            print(f"  Loaded {len(candidates):,} candidates...")
                    except Exception as e:
                        errors += 1
                        if errors < 5:  # Only print first few errors
                            print(f"  Warning: Parse error on line {line_num}: {e}")
                        continue

    except FileNotFoundError:
        print(f"Error: Candidate file not found at '{args.candidates}'")
        sys.exit(1)
    except OSError as e:
        print(f"Error reading candidates file: {e}")
        sys.exit(1)

    print(f"✓ Loaded {len(candidates):,} candidates ({errors} parse errors)\n")

    if not candidates:
        print("Error: No valid candidates found")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # RANKING LOGIC - Choose mode
    # ──────────────────────────────────────────────────────────────────────────

    if args.mode == "fast":
        # Original fast scoring (no semantic, no LLM)
        print("Running FAST mode (original rule-based scoring)...\n")

        scored = []
        for i, candidate in enumerate(candidates):
            try:
                score, breakdown = score_candidate(candidate)
                scored.append((score, candidate, breakdown))

                if (i + 1) % 10000 == 0:
                    print(f"  Scored {i+1:,} candidates...")
            except Exception as e:
                if len(scored) < 5:
                    print(f"  Warning: Scoring error for candidate {i}: {e}")
                continue

        print(f"✓ Scored {len(scored):,} candidates\n")
        print("Sorting...")

        # Sort by score descending, then candidate_id ascending as tie-breaker
        scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))

        top = scored[:args.top]

    elif args.mode == "hybrid":
        # Hybrid ranking: fast filter + semantic/LLM rerank
        from app.services.hybrid_ranker import hybrid_ranking_pipeline
        from app.services.jd_parser import get_jd_requirements

        # Load JD requirements
        jd_reqs = None
        if args.jd and os.path.exists(args.jd):
            print(f"Parsing job description from {args.jd}...\n")
            jd_reqs = get_jd_requirements(args.jd, use_llm=True)
        else:
            print("No JD file provided, using default requirements\n")
            jd_reqs = get_jd_requirements(use_llm=False)

        # Run hybrid pipeline
        top = hybrid_ranking_pipeline(
            candidates=candidates,
            jd_requirements=jd_reqs,
            top_n=args.top,
            rerank_pool=args.rerank_pool,
            use_semantic=not args.no_semantic,
            use_llm_career=not args.no_llm_career,
            use_llm_rerank=False,  # Not implemented yet
            llm_provider=args.llm_provider
        )

    elif args.mode == "enhanced":
        # Full enhanced mode (semantic + LLM on all candidates - slow but highest quality)
        print("Running ENHANCED mode (full semantic + LLM on all candidates)...\n")
        print("⚠️  Warning: This mode is SLOW and EXPENSIVE for large datasets!\n")

        from app.services.hybrid_ranker import hybrid_ranking_pipeline
        from app.services.jd_parser import get_jd_requirements

        jd_reqs = None
        if args.jd and os.path.exists(args.jd):
            jd_reqs = get_jd_requirements(args.jd, use_llm=True)
        else:
            jd_reqs = get_jd_requirements(use_llm=False)

        # Use hybrid pipeline but with larger rerank pool
        top = hybrid_ranking_pipeline(
            candidates=candidates,
            jd_requirements=jd_reqs,
            top_n=args.top,
            rerank_pool=min(len(candidates), 2000),  # Larger pool
            use_semantic=not args.no_semantic,
            use_llm_career=not args.no_llm_career,
            use_llm_rerank=False,
            llm_provider=args.llm_provider
        )

    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)
    
    # ──────────────────────────────────────────────────────────────────────────
    # OUTPUT - Write CSV
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\nWriting top {len(top)} candidates to {args.out} ...")

    try:
        with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rank, (score, candidate, breakdown) in enumerate(top, 1):
                cid = candidate["candidate_id"]
                reasoning = generate_reasoning(candidate, breakdown)
                writer.writerow([cid, rank, f"{score:.6f}", reasoning])
    except OSError as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

    print(f"✓ Output written to {args.out}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{'='*80}")
    print("RANKING COMPLETE")
    print(f"{'='*80}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Total candidates processed: {len(candidates):,}")
    print(f"Output: {len(top)} candidates")

    if top:
        print(f"\nTop candidate: {top[0][1]['candidate_id']} (Score: {top[0][0]:.6f})")
        print(f"Score range: {top[0][0]:.6f} → {top[-1][0]:.6f}")

        # Show enhanced features used
        if args.mode in ["hybrid", "enhanced"]:
            sample_breakdown = top[0][2]
            enhanced_features = []
            if "semantic_skill" in sample_breakdown:
                enhanced_features.append("✓ Semantic skill matching")
            if "llm_career" in sample_breakdown:
                enhanced_features.append("✓ LLM career evaluation")

            if enhanced_features:
                print("\nEnhanced features used:")
                for feature in enhanced_features:
                    print(f"  {feature}")

    print(f"{'='*80}\n")
    print("✅ Ranking run complete successfully!")

if __name__ == "__main__":
    main()
