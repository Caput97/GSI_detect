import json, warnings, os, pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, classification_report

# ######################### paths ###################################
GOLD_PATH = "path/to/organisers/file/test_set_key.jsonl"	# path to organisers' file with golden labels
PRED_PATH = "path/to/participants/run/output.jsonl"	# path to participants' file with predicted labels
OUT_PATH  = "path/to/participants/run/evaluation.csv"	# path to output file CSV
# ###################################################################

VALID_CATEGORIES = {"role", "personality", "competence", "physical", "sexual", "relational"}


def load_jsonl(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                if "id" not in item:
                    warnings.warn(f"{path}, line {i}: missing 'id'. Skipped.")
                    continue
                data[item["id"]] = item
            except json.JSONDecodeError as e:
                warnings.warn(f"{path}, line {i}: invalid JSON ({e}). Skipped.")
    return data


def validate_preds(preds):
    valid, malformed = {}, []
    for pid, p in preds.items():
        val, cat = 0.0, ""

        # Validating gs_value
        try:
            value = float(p.get("gs_value", 0.0))
            if 0 <= value <= 1:
                val = value
            else:
                warnings.warn(f"id={pid}: gs_value out of range [0,1], replaced with 0.")
                malformed.append(pid)
        except Exception:
            warnings.warn(f"id={pid}: gs_value missing or invalid, replaced with 0.")
            malformed.append(pid)

        # Validating gs_category
        category = str(p.get("gs_category", "")).lower()
        if category and category != "no":
            if category in VALID_CATEGORIES:
                cat = category
            else:
                warnings.warn(f"id={pid}: gs_category '{category}' invalid, replaced with empty string.")
                malformed.append(pid)
        else:
            cat = ""  # empty or "no" = no category

        valid[pid] = {"id": pid, "text": p.get("text", ""), "gs_value": val, "gs_category": cat}

    return valid, malformed


def concordance_correlation_coefficient(y_true, y_pred):
    #CCC
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true, ddof=0)
    var_pred = np.var(y_pred, ddof=0)
    cov = np.cov(y_true, y_pred, ddof=0)[0, 1]

    numerator = 2 * cov
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator if denominator != 0 else 0


def evaluate(gold, pred):
    ids = set(gold) & set(pred)
    if not ids:
        raise ValueError("No overlapping IDs found.")

    y_true_val = [float(gold[i]["gs_value"]) for i in ids]
    y_pred_val = [float(pred[i]["gs_value"]) for i in ids]
    
    mse = mean_squared_error(y_true_val, y_pred_val)
    mse_score = max(0.0, 1.0 - mse)
    ccc = concordance_correlation_coefficient(y_true_val, y_pred_val)

    var_true = np.var(y_true_val)
    nmse = mse / var_true if var_true > 0 else float("nan")
    nmse_score = 1 / (1 + nmse) if nmse == nmse and nmse >= 0 else float("nan")

    y_true_cat, y_pred_cat = [], []
    excluded_ids = []
    for i in ids:
        gcat = str(gold[i].get("gs_category", "")).lower()
        if gcat in VALID_CATEGORIES:
            y_true_cat.append(gcat)
            y_pred_cat.append(pred[i]["gs_category"])
        #else:
            # logghiamo l'id escluso + il valore reale della categoria gold
            #excluded_ids.append({
            #    "id": i,
            #    "gold_category": gcat
            #})
    excluded = len(ids) - len(y_true_cat)

    if y_true_cat:
        f1_macro = f1_score(y_true_cat, y_pred_cat, average="macro")
        f1_micro = f1_score(y_true_cat, y_pred_cat, average="micro")
        report = classification_report(y_true_cat, y_pred_cat, digits=6, output_dict=True)
    else:
        f1_macro = f1_micro = 0.0
        report = {}

    return {
        "mse": mse,
        "mse_score": mse_score,
        "nmse": nmse,
        "nmse_score": nmse_score,
        "ccc": ccc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "report": report,
        "excluded": excluded,
        #"excluded_ids": excluded_ids,
        "total": len(ids)
    }


def save_results(results, malformed, out_path, summary_text):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # column structure
    columns = [
        "metric_type", "category", "precision", "recall", "f1_score", "support",
        "mse", "nmse", "nmse_score", "mse_score", "ccc",
        "f1_macro", "f1_micro", "excluded", "malformed_predictions"
    ]

    # gs_value metrics
    value_metrics = pd.DataFrame([{
        "metric_type": "gs_value",
        "category": "overall",
        "precision": "",
        "recall": "",
        "f1_score": "",
        "support": "",
        "mse": results["mse"],
        "nmse": results["nmse"],
        "mse_score": results["mse_score"],
        "nmse_score": results["nmse_score"],
        "ccc": results["ccc"],
        "f1_macro": "",
        "f1_micro": "",
        "excluded": "",
        "malformed_predictions": len(malformed),
    }], columns=columns)

    # gs_category metrics
    cat_rows = []
    if results["report"]:
        for cat, vals in results["report"].items():
            if cat in VALID_CATEGORIES:
                cat_rows.append({
                    "metric_type": "gs_category",
                    "category": cat,
                    "precision": vals["precision"],
                    "recall": vals["recall"],
                    "f1_score": vals["f1-score"],
                    "support": vals["support"],
                    "mse": "",
                    "nmse": "",
                    "mse_score": "",
                    "ccc": "",
                    "f1_macro": "",
                    "f1_micro": "",
                    "excluded": "",
                    "malformed_predictions": "",
                })

    cat_rows.append({
        "metric_type": "gs_category_overall",
        "category": "overall",
        "precision": "",
        "recall": "",
        "f1_score": "",
        "support": "",
        "mse": "",
        "nmse": "",
        "mse_score": "",
        "ccc": "",
        "f1_macro": results["f1_macro"],
        "f1_micro": results["f1_micro"],
        "excluded": results["excluded"],
        "malformed_predictions": len(malformed),
    })

    cat_metrics = pd.DataFrame(cat_rows, columns=columns)

    # writing CSV file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("### VALUE-LEVEL METRICS ###\n")
        value_metrics.to_csv(f, index=False)
        f.write("\n\n### CATEGORY-LEVEL METRICS ###\n")
        cat_metrics.to_csv(f, index=False)

    # writing TXT file
    txt_path = os.path.splitext(out_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nResults saved to:\n  CSV: {out_path}\n  TXT: {txt_path}")


def main():
    print(f"Loading gold labels from: {GOLD_PATH}")
    print(f"Loading predicted labels from: {PRED_PATH}")
    gold = load_jsonl(GOLD_PATH)
    raw_pred = load_jsonl(PRED_PATH)
    pred, malformed = validate_preds(raw_pred)
    res = evaluate(gold, pred)

    # --- STRUCTURED SUMMARY ---
    summary = (
        f"\n=== Evaluation Results ===\n"
        f"Samples: {res['total']}\n"
        f"Invalid predictions replaced: {len(malformed)}\n"
        f"Excluded from F1 (no valid gold category): {res['excluded']}\n"
        "\n--- GS_VALUE METRICS ---\n"
        f"NMSE Score (1/(1+NMSE)): {res['nmse_score']:.4f}\n"
        f"MSE (error): {res['mse']:.4f}\n"
        f"NMSE (normalized): {res['nmse']:.4f}\n"
        f"MSE Score Inverted (1 - MSE): {res['mse_score']:.4f}\n"
        f"CCC (Concordance Corr. Coeff.): {res['ccc']:.4f}\n"
        "\n--- GS_CATEGORY METRICS ---\n"
        f"F1 Macro: {res['f1_macro']:.4f}\n"
        f"F1 Micro: {res['f1_micro']:.4f}\n"
        )

    print(summary)

    print()
    print()
    print()

    '''
    # --- lista esclusi ---
    summary += "\n--- EXCLUDED IDS (no valid gold category) ---\n"

    for item in res["excluded_ids"]:
        summary += f"ID={item['id']}, GOLD_CATEGORY='{item['gold_category']}'\n"
    '''
    # Per-category section
    if res["report"]:
        summary += "\n--- Per-Category F1 ---\n"
        print("\n--- Per-Category F1 ---")
        for cat, vals in res["report"].items():
            if cat in VALID_CATEGORIES:
                line = f"{cat:<12}: F1={vals['f1-score']:.4f} (P={vals['precision']:.4f}, R={vals['recall']:.4f})"
                summary += line + "\n"
                print(line)

    save_results(res, malformed, OUT_PATH, summary)


if __name__ == "__main__":
    main()
