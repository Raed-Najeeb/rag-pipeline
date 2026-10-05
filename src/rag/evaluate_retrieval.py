import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieve import retrieve

TEST_CASES_PATH = "data/test_cases/retrieval_tests.jsonl"


def load_test_cases(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def evaluate(n_results=3):
    cases = load_test_cases(TEST_CASES_PATH)
    results = []

    for case in cases:
        retrieved = retrieve(case["question"], n_results=n_results)
        retrieved_docs = [r["doc"] for r in retrieved]

        if case["expected_doc"] is None:
            correct = True
            note = f"out-of-scope question; top retrieved: {retrieved_docs[0]}"
        else:
            correct = case["expected_doc"] in retrieved_docs
            note = f"expected {case['expected_doc']}, got {retrieved_docs}"

        results.append({
            "id": case["id"],
            "question": case["question"],
            "correct": correct,
            "note": note,
        })
        status = "PASS" if correct else "FAIL"
        print(f"[{status}] {case['id']}: {case['question']}")
        print(f"       {note}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nRetrieval accuracy: {accuracy:.0%} ({sum(r['correct'] for r in results)}/{len(results)})")
    return results


if __name__ == "__main__":
    evaluate()