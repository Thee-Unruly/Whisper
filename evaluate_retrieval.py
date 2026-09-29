"""
Evaluation module for Information Retrieval (IR) / RAG pipeline.
Calculates Precision, Recall, F1-Score, Hit Rate, and MRR (Mean Reciprocal Rank)
for retrieved transcript chunks against a ground truth dataset.

Usage:
  python evaluate_retrieval.py --ground-truth ground_truth.json --top-k 5
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Import embedding lookup logic from transcribe_to_kb
from transcribe_to_kb import get_embeddings, get_db_connection

def retrieve_top_k(query: str, top_k: int = 5, client: str = None, module: str = None) -> List[Dict[str, Any]]:
    """Retrieves top_k document chunks from PostgreSQL vector store."""
    query_emb = get_embeddings([query])[0]
    q_literal = "[" + ",".join(str(x) for x in query_emb) + "]"

    conn = get_db_connection()
    cur = conn.cursor()

    where_clauses = []
    params = [q_literal]

    if client:
        where_clauses.append("source LIKE %s")
        params.append(f"%client:{client}%")
    if module:
        where_clauses.append("source LIKE %s")
        params.append(f"%module:{module}%")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.append(top_k)

    sql = f"""
        SELECT id, source, chunk_text, (embedding <=> %s::vector) AS distance
        FROM public.documents
        {where_sql}
        ORDER BY distance ASC
        LIMIT %s;
    """

    cur.execute(sql, tuple(params))
    results = cur.fetchall()
    cur.close()
    conn.close()

    retrieved = []
    for r in results:
        doc_id, source, text, dist = r
        retrieved.append({
            "id": doc_id,
            "source": source,
            "text": text,
            "distance": float(dist),
            "score": max(0.0, 1.0 - float(dist))
        })
    return retrieved


def is_match(doc: Dict[str, Any], relevant_spec: Any) -> bool:
    """Checks if a retrieved document chunk matches a ground truth specification (ID, filename, text snippet, or keyword)."""
    if isinstance(relevant_spec, int):
        return doc["id"] == relevant_spec
    
    spec_str = str(relevant_spec).strip().lower()
    if not spec_str:
        return False
        
    # Check ID string
    if str(doc.get("id")) == spec_str:
        return True
        
    # Check in source metadata (filename, client, module)
    source_str = str(doc.get("source", "")).lower()
    if spec_str in source_str:
        return True
        
    # Check in chunk text
    chunk_text = str(doc.get("text", "")).lower()
    if spec_str in chunk_text:
        return True

    return False


def calculate_metrics_flexible(relevant_items: List[Any], retrieved_docs: List[Dict[str, Any]], k: int) -> Dict[str, float]:
    """
    Computes IR metrics for a single query supporting doc IDs, filenames, or keywords:
    - Precision@k
    - Recall@k
    - F1-Score@k
    - Hit Rate@k
    - MRR (Mean Reciprocal Rank)
    """
    retrieved_k = retrieved_docs[:k]
    if not relevant_items:
        return {
            "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "hit_rate": 0.0, "mrr": 0.0
        }

    # Identify retrieved docs that match at least one expected item
    matched_retrieved_indices = set()
    matched_relevant_specs = set()

    for r_idx, doc in enumerate(retrieved_k):
        for spec_idx, spec in enumerate(relevant_items):
            if is_match(doc, spec):
                matched_retrieved_indices.add(r_idx)
                matched_relevant_specs.add(spec_idx)

    tp = len(matched_retrieved_indices)
    precision = tp / k if k > 0 else 0.0
    recall = len(matched_relevant_specs) / len(relevant_items) if len(relevant_items) > 0 else 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    hit_rate = 1.0 if tp > 0 else 0.0

    # Reciprocal Rank (RR)
    reciprocal_rank = 0.0
    for rank, doc in enumerate(retrieved_k, 1):
        if any(is_match(doc, spec) for spec in relevant_items):
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "hit_rate": hit_rate,
        "mrr": reciprocal_rank
    }


def _band(value: float) -> str:
    if value >= 0.8: return "strong"
    if value >= 0.5: return "moderate"
    if value >= 0.25: return "weak"
    return "very weak"


def explain_metrics(summary: Dict[str, Any], details: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Turns the numeric scores into plain-language explanations."""
    k = summary["top_k"]
    n = summary["total_queries"]
    p = summary["mean_precision"]
    r = summary["mean_recall"]
    f1 = summary["mean_f1_score"]
    hit = summary["mean_hit_rate"]
    mrr = summary["mrr"]
    single = n == 1 and details

    # Precision
    if single:
        hits = round(p * k)
        precision = (f"Out of the {k} chunks returned, {hits} {'was' if hits == 1 else 'were'} relevant "
                     f"({hits} ÷ {k}). The other {k - hits} were off-target noise.")
    else:
        precision = (f"On average, {p * k:.1f} of every {k} returned chunks were relevant. "
                     f"Low precision means results are padded with unrelated content.")

    # Recall
    if single:
        expected = len(details[0].get("expected", []))
        found = round(r * expected) if expected else 0
        recall = (f"You expected {expected} item(s) to be found and the search found {found} "
                  f"({found} ÷ {expected}). Low recall means relevant material was missed.")
    else:
        recall = (f"On average, the search found {r * 100:.0f}% of the items you expected. "
                  f"Low recall means relevant material is being missed.")

    # F1
    f1_text = (f"The balance between precision and recall. At {f1 * 100:.1f}%, overall retrieval "
               f"quality is {_band(f1)}.")

    # Hit rate
    if single:
        hit_text = ("At least one relevant chunk appeared in the top results, so the system found the right topic."
                    if hit >= 1 else
                    "No relevant chunk appeared in the top results, so the system missed the topic entirely.")
    else:
        hit_text = f"{hit * 100:.0f}% of prompts returned at least one relevant chunk in the top {k}."

    # MRR
    if single and mrr > 0:
        rank = round(1 / mrr)
        mrr_text = (f"The first relevant result appeared at Rank #{rank}. "
                    f"{'This is the best possible score.' if rank == 1 else 'The closer to 1.0, the better.'}")
    elif mrr == 0:
        mrr_text = "No relevant result was found in the top results."
    else:
        mrr_text = (f"On average the first relevant chunk appears around rank {1 / mrr:.1f}. "
                    f"1.0 means the right answer is always first.")

    # Overall verdict
    if hit >= 0.8 and p < 0.5:
        overall = ("The search usually finds the right answer and ranks it well, but it pads the results "
                   "with unrelated chunks and misses other relevant ones.")
    elif hit < 0.5:
        overall = "The search is frequently missing the topic entirely. Check chunking, embeddings, or the filters used."
    elif p >= 0.5 and r >= 0.5:
        overall = "Retrieval is performing well: results are mostly relevant and most expected content is found."
    else:
        overall = "Retrieval is partially working, with room to improve precision and recall."

    caveat = ("Scores depend on how 'relevant' is defined. A chunk counts as a hit only if it contains your "
              "expected keyword/filename/ID, so topically relevant chunks without that exact term are marked off-target.")
    if n < 10:
        caveat += f" Only {n} prompt(s) were tested; 10-20 gives a more reliable picture."

    return {
        "precision": precision, "recall": recall, "f1_score": f1_text,
        "hit_rate": hit_text, "mrr": mrr_text, "overall": overall, "caveat": caveat,
    }




def create_sample_ground_truth(file_path: str):
    """Generates a template ground truth JSON file for evaluation."""
    sample_data = [
        {
            "query": "How to set up accounting period?",
            "relevant_doc_ids": [1, 2],
            "client": "TMRC",
            "module": "02. Finance"
        },
        {
            "query": "What is the process for e-recruitment approval?",
            "relevant_doc_ids": [5],
            "client": "TMRC",
            "module": "03. E-Recruitment"
        }
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=4)
    print(f"Sample ground truth template created at: {file_path}")


def evaluate_pipeline(ground_truth_path: str, top_k: int = 5) -> Dict[str, Any]:
    """Runs retrieval evaluation on ground truth queries."""
    if not os.path.exists(ground_truth_path):
        print(f"Ground truth file not found: {ground_truth_path}")
        print("Generating a sample template ground_truth.json...")
        create_sample_ground_truth(ground_truth_path)
        print("Please update ground_truth.json with actual relevant document IDs from your database.")
        sys.exit(1)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total_queries = len(test_cases)
    if total_queries == 0:
        print("Ground truth dataset is empty.")
        return {}

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_hit_rate = 0.0
    sum_mrr = 0.0

    print(f"\n=======================================================")
    print(f" Running Retrieval Evaluation (Total Queries: {total_queries}, Top K: {top_k})")
    print(f"=======================================================\n")

    query_details = []

    for idx, item in enumerate(test_cases, 1):
        query = item["query"]
        relevant_ids = item.get("relevant_doc_ids", [])
        client = item.get("client")
        module = item.get("module")

        retrieved = retrieve_top_k(query, top_k=top_k, client=client, module=module)
        retrieved_ids = [doc["id"] for doc in retrieved]

        metrics = calculate_metrics_flexible(relevant_ids, retrieved, top_k)

        sum_precision += metrics["precision"]
        sum_recall += metrics["recall"]
        sum_f1 += metrics["f1_score"]
        sum_hit_rate += metrics["hit_rate"]
        sum_mrr += metrics["mrr"]

        query_details.append({
            "query": query,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "metrics": metrics
        })

        print(f"[{idx}/{total_queries}] Query: '{query}'")
        print(f"   Precision@{top_k}: {metrics['precision']:.4f} | Recall@{top_k}: {metrics['recall']:.4f} | F1-Score@{top_k}: {metrics['f1_score']:.4f}")
        print(f"   Hit Rate@{top_k}: {metrics['hit_rate']:.4f} | Reciprocal Rank: {metrics['mrr']:.4f}\n")

    # Aggregate Means
    mean_metrics = {
        "mean_precision": sum_precision / total_queries,
        "mean_recall": sum_recall / total_queries,
        "mean_f1_score": sum_f1 / total_queries,
        "mean_hit_rate": sum_hit_rate / total_queries,
        "mrr": sum_mrr / total_queries,
        "total_queries": total_queries,
        "top_k": top_k
    }

    print("=======================================================")
    print(" EVALUATION SUMMARY RESULTS")
    print("=======================================================")
    print(f" Mean Precision@{top_k}:  {mean_metrics['mean_precision'] * 100:.2f}%")
    print(f" Mean Recall@{top_k}:     {mean_metrics['mean_recall'] * 100:.2f}%")
    print(f" Mean F1-Score@{top_k}:   {mean_metrics['mean_f1_score'] * 100:.2f}%")
    print(f" Mean Hit Rate@{top_k}:   {mean_metrics['mean_hit_rate'] * 100:.2f}%")
    print(f" MRR (Mean Reciprocal Rank): {mean_metrics['mrr']:.4f}")
    print("=======================================================\n")

    return mean_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Information Retrieval (IR) metrics on Whisper transcription chunks.")
    parser.add_argument("--ground-truth", default="ground_truth.json", help="Path to ground truth JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K documents to evaluate")
    parser.add_argument("--create-template", action="store_true", help="Create a sample ground_truth.json template file")

    args = parser.parse_args()

    if args.create_template:
        create_sample_ground_truth(args.ground_truth)
    else:
        evaluate_pipeline(args.ground_truth, top_k=args.top_k)
