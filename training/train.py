"""
train_clinc.py
--------------
Trains the intent classifier using the CLINC150 dataset.
Maps 150 real-world intents → 8 actionable support buckets.

Run from project root:
    python training/train_clinc.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ── Mapping: 150 CLINC intents → your 8 buckets ─────────────
CLINC_TO_INTENT = {
    # REFUND REQUEST
    "dispute_charge":           "refund_request",
    "bill_balance":             "refund_request",
    "refund":                   "refund_request",
    "chargeback":               "refund_request",
    "transaction":              "refund_request",

    # PAYMENT FAILED
    "pay_bill":                 "payment_failed",
    "bill_due":                 "payment_failed",
    "payment":                  "payment_failed",
    "credit_limit":             "payment_failed",
    "credit_limit_change":      "payment_failed",
    "credit_score":             "payment_failed",
    "international_fees":       "payment_failed",
    "transfer":                 "payment_failed",
    "transfer_charge":          "payment_failed",
    "check_balance":            "payment_failed",

    # ACCOUNT SUSPENDED
    "freeze_account":           "account_suspended",
    "pin_change":               "account_suspended",
    "lost_card":                "account_suspended",
    "replacement_card_due":     "account_suspended",
    "expiration_date":          "account_suspended",
    "damaged_card":             "account_suspended",
    "card_declined":            "account_suspended",
    "account_blocked":          "account_suspended",

    # API / TECHNICAL HELP
    "reset_settings":           "api_integration_help",
    "sync_device":              "api_integration_help",
    "setup_2fa":                "api_integration_help",
    "change_language":          "api_integration_help",
    "change_user_name":         "api_integration_help",
    "change_password":          "api_integration_help",
    "change_speed":             "api_integration_help",
    "change_volume":            "api_integration_help",
    "iot_cleaning":             "api_integration_help",
    "iot_coffee":               "api_integration_help",
    "iot_hue_lightbulb":        "api_integration_help",
    "iot_wemo_off":             "api_integration_help",
    "iot_wemo_on":              "api_integration_help",
    "update_playlist":          "api_integration_help",
    "plug_type":                "api_integration_help",
    "wifi_password":            "api_integration_help",
    "calendar":                 "api_integration_help",
    "calendar_update":          "api_integration_help",
    "reminder":                 "api_integration_help",
    "reminder_update":          "api_integration_help",
    "alarm":                    "api_integration_help",
    "timer":                    "api_integration_help",

    # SUBSCRIPTION UPGRADE
    "upgrade":                  "subscription_upgrade",
    "insurance":                "subscription_upgrade",
    "insurance_change":         "subscription_upgrade",
    "rewards_balance":          "subscription_upgrade",
    "rollover_days":            "subscription_upgrade",
    "apr":                      "subscription_upgrade",
    "credit_limit_change":      "subscription_upgrade",

    # SUBSCRIPTION CANCEL
    "cancel":                   "subscription_cancel",
    "cancel_reservation":       "subscription_cancel",
    "pto_request":              "subscription_cancel",
    "pto_request_status":       "subscription_cancel",
    "pto_balance":              "subscription_cancel",
    "payday":                   "subscription_cancel",
    "direct_deposit":           "subscription_cancel",

    # SECURITY CONCERN
    "report_fraud":             "security_concern",
    "compromised_acct":         "security_concern",
    "stolen_card":              "security_concern",
    "identity_theft":           "security_concern",
    "phishing_email":           "security_concern",
    "suspicious_activity":      "security_concern",
    "privacy":                  "security_concern",
    "2fa_question":             "security_concern",
    "account_security":         "security_concern",

    # GENERAL INQUIRY
    "routing":                  "general_inquiry",
    "interest_rate":            "general_inquiry",
    "min_payment":              "general_inquiry",
    "order_status":             "general_inquiry",
    "order_checks":             "general_inquiry",
    "spending_history":         "general_inquiry",
    "account_opening":          "general_inquiry",
    "application_status":       "general_inquiry",
    "what_is_your_name":        "general_inquiry",
    "who_made_you":             "general_inquiry",
    "where_are_you_from":       "general_inquiry",
    "how_old_are_you":          "general_inquiry",
    "are_you_a_bot":            "general_inquiry",
    "meaning_of_life":          "general_inquiry",
    "fun_fact":                 "general_inquiry",
    "tell_joke":                "general_inquiry",
    "hours":                    "general_inquiry",
    "contact":                  "general_inquiry",
    "greeting":                 "general_inquiry",
    "goodbye":                  "general_inquiry",
    "thank_you":                "general_inquiry",
    "yes":                      "general_inquiry",
    "no":                       "general_inquiry",
    "repeat":                   "general_inquiry",
    "user_name":                "general_inquiry",
    "income":                   "general_inquiry",
    "taxes":                    "general_inquiry",
    "w2":                       "general_inquiry",
    "distance":                 "general_inquiry",
    "flip_coin":                "general_inquiry",
    "roll_dice":                "general_inquiry",
    "gas":                      "general_inquiry",
    "gas_type":                 "general_inquiry",
    "oil_change":               "general_inquiry",
    "jump_start":               "general_inquiry",
    "uber":                     "general_inquiry",
    "restaurant_reviews":       "general_inquiry",
    "restaurant_reservation":   "general_inquiry",
    "accept_reservations":      "general_inquiry",
    "calories":                 "general_inquiry",
    "nutrition_info":           "general_inquiry",
    "traffic":                  "general_inquiry",
    "directions":               "general_inquiry",
    "flight_status":            "general_inquiry",
    "travel_alert":             "general_inquiry",
    "travel_suggestion":        "general_inquiry",
    "translate":                "general_inquiry",
    "definition":               "general_inquiry",
    "spelling":                 "general_inquiry",
    "weather":                  "general_inquiry",
    "forecast":                 "general_inquiry",
    "current_location":         "general_inquiry",
    "share_location":           "general_inquiry",
    "book_hotel":               "general_inquiry",
    "book_flight":              "general_inquiry",
    "next_song":                "general_inquiry",
    "play_music":               "general_inquiry",
    "music_settings":           "general_inquiry",
    "todo_list":                "general_inquiry",
    "todo_list_update":         "general_inquiry",
    "shopping_list":            "general_inquiry",
    "shopping_list_update":     "general_inquiry",
    "what_song":                "general_inquiry",
    "who_sings":                "general_inquiry",
    "text":                     "general_inquiry",
    "make_call":                "general_inquiry",
    "redial":                   "general_inquiry",
    "date":                     "general_inquiry",
    "calculator":               "general_inquiry",
    "measurement_conversion":   "general_inquiry",
    "unit_conversion":          "general_inquiry",
    "meeting_schedule":         "general_inquiry",
    "carry_on":                 "general_inquiry",
    "vaccines":                 "general_inquiry",
    "visa":                     "general_inquiry",
    "exchange_rate":            "general_inquiry",
    "international_visa":       "general_inquiry",
    "timezone":                 "general_inquiry",
    "find_phone":               "general_inquiry",
    "smart_home":               "general_inquiry",
    "schedule_maintenance":     "general_inquiry",
    "time_zone":                "general_inquiry",
}


def load_clinc_split(ds, split):
    """Extract texts and map CLINC intent IDs → your 8 bucket labels."""
    intent_names = ds[split].features["intent"].names
    texts, labels = [], []
    skipped = 0
    for row in ds[split]:
        clinc_name = intent_names[row["intent"]]
        if clinc_name == "oos":          # out-of-scope — skip
            skipped += 1
            continue
        mapped = CLINC_TO_INTENT.get(clinc_name)
        if mapped is None:               # unmapped intent — skip
            skipped += 1
            continue
        texts.append(row["text"])
        labels.append(mapped)
    return texts, labels, skipped


print("=" * 62)
print("  INTENT CLASSIFIER — CLINC150 TRAINING PIPELINE")
print("=" * 62)

# ── 1. Load CLINC150 ─────────────────────────────────────────
print("\n[1/5] Loading CLINC150 dataset from HuggingFace...")
ds = load_dataset("clinc_oos", "plus")
print(f"  Train rows (raw): {len(ds['train'])}")
print(f"  Test rows  (raw): {len(ds['test'])}")

# ── 2. Map intents ───────────────────────────────────────────
print("\n[2/5] Mapping 150 CLINC intents → 8 buckets...")
X_train, y_train, skipped_train = load_clinc_split(ds, "train")
X_test,  y_test,  skipped_test  = load_clinc_split(ds, "test")

print(f"  Train samples kept: {len(X_train)}  (skipped {skipped_train})")
print(f"  Test  samples kept: {len(X_test)}   (skipped {skipped_test})")
print()
counts = Counter(y_train)
for intent, n in sorted(counts.items()):
    print(f"    {intent:<25} {n:>4} train samples")

# ── 3. Encode labels ─────────────────────────────────────────
print("\n[3/5] Encoding labels...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
print(f"  Classes: {list(le.classes_)}")

# ── 4. Generate embeddings ───────────────────────────────────
print("\n[4/5] Generating sentence embeddings...")
print("  Model: all-MiniLM-L6-v2  (downloads ~80MB on first run)")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("  Encoding training set...")
X_train_emb = embedder.encode(X_train, show_progress_bar=True, batch_size=64)

print("  Encoding test set...")
X_test_emb  = embedder.encode(X_test,  show_progress_bar=True, batch_size=64)

print(f"  Embedding shape: {X_train_emb.shape}")

# ── 5. Train classifier ──────────────────────────────────────
print("\n[5/5] Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
clf.fit(X_train_emb, y_train_enc)
print("  Done!")

# ── Evaluation ───────────────────────────────────────────────
print("\n" + "=" * 62)
print("  EVALUATION  (on real held-out CLINC test set)")
print("=" * 62)

y_pred = clf.predict(X_test_emb)
acc = accuracy_score(y_test_enc, y_pred)

print(f"\n  Overall Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test_enc, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("Confusion Matrix:")
print(cm_df.to_string())

# ── Save ─────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  SAVING MODELS")
print("=" * 62)

joblib.dump(clf,      "models/intent_classifier.joblib")
joblib.dump(le,       "models/label_encoder.joblib")
joblib.dump(embedder, "models/embedder")

meta = {
    "classes": list(le.classes_),
    "accuracy": round(float(acc), 4),
    "embedding_model": "all-MiniLM-L6-v2",
    "training_data": "CLINC150 (23,700 real utterances → mapped to 8 buckets)",
    "num_train_samples": len(X_train),
    "num_test_samples": len(X_test),
    "intents": {
        cls: {"description": desc, "route_to": route, "color": color}
        for cls, desc, route, color in [
            ("refund_request",       "User wants money back",             "Billing Team",        "#f59e0b"),
            ("payment_failed",       "Payment is not going through",      "Payment Support",     "#ef4444"),
            ("account_suspended",    "Account locked or banned",          "Trust & Safety",      "#8b5cf6"),
            ("api_integration_help", "Developer needs API/SDK help",      "Developer Support",   "#3b82f6"),
            ("subscription_upgrade", "User wants to upgrade their plan",  "Sales Team",          "#10b981"),
            ("subscription_cancel",  "User wants to cancel",              "Retention Team",      "#f97316"),
            ("security_concern",     "Security breach or fraud",          "Security Team",       "#dc2626"),
            ("general_inquiry",      "General question or information",   "General Support",     "#6b7280"),
        ]
    }
}

with open("models/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("  intent_classifier.joblib saved")
print("  label_encoder.joblib saved")
print("  embedder/ saved")
print("  metadata.json saved")

# ── Quick inference test ─────────────────────────────────────
print("\n" + "=" * 62)
print("  QUICK INFERENCE TEST")
print("=" * 62)

test_msgs = [
    "I want a refund for my last charge",
    "my card keeps getting declined",
    "how do I authenticate API requests",
    "i think my account was hacked",
    "please cancel my subscription",
    "I want to upgrade to Pro",
    "my account is suspended what do i do",
    "ugh why wont it let me pay",         # messy real-world input
    "i got locked out lol",               # slang
    "bruh just take my money already",    # very casual
]

X_inf = embedder.encode(test_msgs)
preds = clf.predict(X_inf)
probs = clf.predict_proba(X_inf)

print(f"\n{'Message':<42} {'Intent':<25} {'Conf':>6}  Tier")
print("-" * 85)
for msg, pred, prob in zip(test_msgs, preds, probs):
    intent = le.inverse_transform([pred])[0]
    conf = prob.max()
    tier = "high" if conf >= 0.85 else "medium" if conf >= 0.60 else "low"
    print(f"{msg:<42} {intent:<25} {conf*100:>5.1f}%  [{tier}]")

print(f"\n Training complete! Real-world accuracy: {acc*100:.2f}%")