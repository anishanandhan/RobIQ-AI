#!/usr/bin/env python3
"""
Compare original vs enhanced rankings to show improvements.
"""

import csv
import sys

def load_rankings(file_path):
    """Load rankings from CSV file."""
    rankings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['candidate_id']
            rankings[cid] = {
                'rank': int(row['rank']),
                'score': float(row['score']),
                'reasoning': row['reasoning']
            }
    return rankings

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_rankings.py original.csv enhanced.csv")
        sys.exit(1)

    original_file = sys.argv[1]
    enhanced_file = sys.argv[2]

    print("Loading rankings...")
    original = load_rankings(original_file)
    enhanced = load_rankings(enhanced_file)

    print(f"\n{'='*100}")
    print("RANKING COMPARISON: Original vs Enhanced (with Semantic Matching)")
    print(f"{'='*100}\n")

    # Score statistics
    orig_scores = [v['score'] for v in original.values()]
    enh_scores = [v['score'] for v in enhanced.values()]

    print("Score Statistics:")
    print(f"  Original: {min(orig_scores):.4f} → {max(orig_scores):.4f} (mean: {sum(orig_scores)/len(orig_scores):.4f})")
    print(f"  Enhanced: {min(enh_scores):.4f} → {max(enh_scores):.4f} (mean: {sum(enh_scores)/len(enh_scores):.4f})")
    print(f"  Improvement: +{(max(enh_scores) - max(orig_scores))/max(orig_scores)*100:.1f}% on top score")
    print()

    # Top 10 comparison
    print("Top 10 Candidates Comparison:")
    print(f"{'Rank':>5} | {'Original ID':^15} | {'Score':>7} | {'Enhanced ID':^15} | {'Score':>7} | {'Change':>6}")
    print(f"{'-'*5}-+-{'-'*15}-+-{'-'*7}-+-{'-'*15}-+-{'-'*7}-+-{'-'*6}")

    orig_sorted = sorted(original.items(), key=lambda x: x[1]['rank'])
    enh_sorted = sorted(enhanced.items(), key=lambda x: x[1]['rank'])

    for i in range(10):
        orig_cid, orig_data = orig_sorted[i]
        enh_cid, enh_data = enh_sorted[i]

        # Check if candidate moved ranks
        if orig_cid == enh_cid:
            change = "="
        elif orig_cid in enhanced:
            orig_in_enh = enhanced[orig_cid]['rank']
            change = f"↓{orig_in_enh - (i+1)}" if orig_in_enh > (i+1) else f"↑{(i+1) - orig_in_enh}"
        else:
            change = "NEW"

        print(f"{i+1:>5} | {orig_cid:^15} | {orig_data['score']:>7.4f} | {enh_cid:^15} | {enh_data['score']:>7.4f} | {change:>6}")

    print()

    # New entries in top 20
    orig_top20 = set(cid for cid, data in orig_sorted[:20])
    enh_top20 = set(cid for cid, data in enh_sorted[:20])

    new_in_top20 = enh_top20 - orig_top20
    dropped_from_top20 = orig_top20 - enh_top20

    if new_in_top20:
        print("New candidates in enhanced top 20:")
        for cid in sorted(new_in_top20):
            enh_rank = enhanced[cid]['rank']
            orig_rank = original.get(cid, {}).get('rank', 'N/A')
            print(f"  {cid}: #{enh_rank} (was #{orig_rank})")
        print()

    if dropped_from_top20:
        print("Dropped from top 20 in enhanced:")
        for cid in sorted(dropped_from_top20):
            orig_rank = original[cid]['rank']
            enh_rank = enhanced.get(cid, {}).get('rank', 'N/A')
            print(f"  {cid}: was #{orig_rank}, now #{enh_rank}")
        print()

    # Biggest movers
    movers = []
    for cid in set(original.keys()) & set(enhanced.keys()):
        orig_rank = original[cid]['rank']
        enh_rank = enhanced[cid]['rank']
        diff = orig_rank - enh_rank
        if diff != 0:
            movers.append((cid, orig_rank, enh_rank, diff))

    if movers:
        movers.sort(key=lambda x: abs(x[3]), reverse=True)

        print("Biggest rank changes (top 10):")
        print(f"{'Candidate':^15} | {'Original':>9} | {'Enhanced':>9} | {'Change':>8}")
        print(f"{'-'*15}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
        for cid, orig_rank, enh_rank, diff in movers[:10]:
            arrow = "↑" if diff > 0 else "↓"
            print(f"{cid:^15} | #{orig_rank:>8} | #{enh_rank:>8} | {arrow}{abs(diff):>7}")

    print(f"\n{'='*100}\n")

if __name__ == "__main__":
    main()
