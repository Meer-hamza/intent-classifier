import json
import joblib
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Mapping: CLINC intents → 8 buckets ──────────────────────
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

# ── Fallback dataset (used if CLINC download fails) ──────────
FALLBACK_DATA = {
    "refund_request": [
        "I want a refund for my last payment",
        "Can I get my money back?",
        "I was charged incorrectly, please refund me",
        "Please refund my subscription",
        "I need my payment reversed",
        "I want to return and get refunded",
        "I accidentally purchased this, can I get a refund?",
        "My refund hasn't been processed yet",
        "I need a full refund immediately",
        "I was double charged, I need a refund",
        "I didn't authorize this charge, refund me",
        "How long does a refund take?",
        "I cancelled but was still charged",
        "Give me back what I paid",
        "I bought the wrong plan, can I get a refund?",
    ],
    "payment_failed": [
        "My payment is not going through",
        "I can't complete my purchase",
        "Payment declined, what do I do?",
        "My card keeps getting rejected",
        "The checkout is failing",
        "Why is my payment failing?",
        "Transaction failed error",
        "My credit card isn't being accepted",
        "Payment not processing",
        "I tried to pay but it's not working",
        "Getting an error when trying to checkout",
        "Card declined but I have funds",
        "Can't pay for my subscription renewal",
        "Payment keeps timing out",
        "Why does my payment keep declining?",
    ],
    "account_suspended": [
        "My account has been suspended",
        "I can't log into my account",
        "My account is locked",
        "I've been banned, why?",
        "Account disabled without reason",
        "Why is my account suspended?",
        "My account got deactivated",
        "I'm locked out of my account",
        "Account restricted unexpectedly",
        "My access has been revoked",
        "I received an account suspension notice",
        "Account blocked suddenly",
        "How do I appeal an account suspension?",
        "I can't log in, it says account suspended",
        "Get my account reinstated",
    ],
    "api_integration_help": [
        "How do I integrate the API?",
        "I need help with the SDK",
        "Webhook not receiving events",
        "API key not working",
        "How to handle API errors?",
        "I'm getting a 401 unauthorized error from the API",
        "How do I authenticate API requests?",
        "API rate limit exceeded",
        "How to test the API in sandbox mode?",
        "I need to integrate payments into my app",
        "How to use the REST API?",
        "Webhook signature verification failing",
        "API response format question",
        "SDK installation for Node.js",
        "I need help debugging my API integration",
    ],
    "subscription_upgrade": [
        "I want to upgrade my plan",
        "How do I move to a higher tier?",
        "I want to switch to the Pro plan",
        "Upgrade my account to enterprise",
        "How to add more seats to my subscription?",
        "I need a plan with more API calls",
        "Can I upgrade mid-cycle?",
        "I want to unlock advanced features",
        "Move me to the business tier",
        "Upgrade from free to paid",
        "How much does the upgrade cost?",
        "I want all the features, how do I upgrade?",
        "Switching from monthly to annual plan",
        "Is there a plan with unlimited usage?",
        "Which plan should I choose?",
    ],
    "subscription_cancel": [
        "I want to cancel my subscription",
        "How do I cancel my plan?",
        "Please cancel my account",
        "I don't want to be charged anymore",
        "Cancel my renewal",
        "How to stop recurring payments?",
        "I want to downgrade to free",
        "Cancel my premium membership",
        "Stop charging my card",
        "I want to end my subscription",
        "How do I turn off auto-renew?",
        "Cancel before the next billing cycle",
        "Please remove me from the subscription",
        "Cancel subscription immediately",
        "I no longer need this service",
    ],
    "security_concern": [
        "I think my account was hacked",
        "Suspicious activity on my account",
        "Someone else is using my account",
        "I didn't make this purchase",
        "Unauthorized login detected",
        "How do I enable two-factor authentication?",
        "My password was compromised",
        "I want to report a security vulnerability",
        "Someone accessed my account without permission",
        "Security breach on my account",
        "I got a login alert I didn't trigger",
        "My API key was exposed",
        "I think there's a data breach",
        "Account takeover attempt",
        "I got a ransom email claiming to have my data",
    ],
    "general_inquiry": [
        "What services do you offer?",
        "How does your platform work?",
        "Can you help me understand pricing?",
        "What payment methods do you support?",
        "Do you support international payments?",
        "What currencies are supported?",
        "Are you PCI compliant?",
        "Do you have a mobile app?",
        "What countries do you operate in?",
        "Can small businesses use your service?",
        "How do I get started?",
        "Is there a free trial available?",
        "Do you offer 24/7 support?",
        "How do I contact support?",
        "What's the difference between your plans?",
    ],
}


def load_clinc():
    """Try to load CLINC150, return None if it fails."""
    try:
        from datasets import load_dataset
        print("  Downloading CLINC150 from HuggingFace...")
        ds = load_dataset("clinc_oos", "plus")
        intent_names = ds["train"].features["intent"].names

        X_train, y_train = [], []
        X_test,  y_test  = [], []

        for row in ds["train"]:
            name = intent_names[row["intent"]]
            mapped = CLINC_TO_INTENT.get(name)
            if mapped:
                X_train.append(row["text"])
                y_train.append(mapped)

        for row in ds["test"]:
            name = intent_names[row["intent"]]
            mapped = CLINC_TO_INTENT.get(name)
            if mapped:
                X_test.append(row["text"])
                y_test.append(mapped)

        print(f"  CLINC loaded: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"  CLINC download failed ({e}) — using fallback dataset")
        return None


def load_fallback():
    """Use our own synthetic dataset as fallback."""
    import random
    random.seed(42)
    X, y = [], []
    for intent, msgs in FALLBACK_DATA.items():
        for msg in msgs:
            X.append(msg)
            y.append(intent)
            X.append(msg.lower())
            y.append(intent)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    split = int(len(X) * 0.8)
    print(f"  Fallback dataset: {split} train, {len(X)-split} test samples")
    return list(X[:split]), list(y[:split]), list(X[split:]), list(y[split:])


print("=" * 60)
print("  INTENT CLASSIFIER — TRAINING PIPELINE")
print("=" * 60)

# 1. Load data
print("\n[1/4] Loading dataset...")
result = load_clinc()
if result:
    X_train, y_train, X_test, y_test = result
    data_source = "CLINC150"
else:
    X_train, y_train, X_test, y_test = load_fallback()
    data_source = "Fallback synthetic dataset"

# 2. Encode labels
print("\n[2/4] Encoding labels...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
print(f"  Classes: {list(le.classes_)}")
counts = Counter(y_train)
for intent, n in sorted(counts.items()):
    print(f"    {intent:<25} {n:>4} samples")

# 3. TF-IDF features (no external model download needed)
print("\n[3/4] Building TF-IDF features...")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"  Feature matrix: {X_train_vec.shape}")

# 4. Train
print("\n[4/4] Training classifier...")
clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
clf.fit(X_train_vec, y_train_enc)

# Evaluate
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test_enc, y_pred)
print(f"\n  Accuracy: {acc*100:.2f}%")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Save
Path("models").mkdir(exist_ok=True)
joblib.dump(clf,        "models/intent_classifier.joblib")
joblib.dump(le,         "models/label_encoder.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

meta = {
    "classes": list(le.classes_),
    "accuracy": round(float(acc), 4),
    "feature_model": "TF-IDF ngram(1,3)",
    "training_data": data_source,
    "num_train_samples": len(X_train),
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

print(f"\n  All models saved! Trained on: {data_source}")