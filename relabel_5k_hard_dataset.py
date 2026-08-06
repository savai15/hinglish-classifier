"""
Semantic Urgency Relabeling Script for 5,000 Hard/Ambiguous Hinglish Complaints

Applies a multi-criteria semantic scoring rule engine to map real complaint text
to High, Medium, or Low urgency based on:
1. Financial Loss / Monetary Value
2. Legal Action / Threat Escalation
3. Severe Operational Failure (Damaged items, Empty box, Fraud)
4. Moderate Operational Delays
5. Informational / FAQ Queries
"""

import os
import re
import pandas as pd

def compute_semantic_urgency(text, category):
    text_lower = text.lower()
    score = 0
    
    # ------------------------------------------------------------------------
    # 1. HIGH URGENCY SIGNALS (+3 points each)
    # ------------------------------------------------------------------------
    high_threat_keywords = [
        'consumer court', 'consumer forum', 'legal action', 'police complaint',
        'fraud', 'scam', 'stole', 'stolen', 'manager ko bulao', 'escalate',
        'immediately', 'turant', 'emergency', 'shattered', 'empty box', 'fake seller'
    ]
    for kw in high_threat_keywords:
        if kw in text_lower:
            score += 3

    # High Risk Categories with financial loss or physical damage
    if category in ['Seller_Fraud', 'Damaged_Product', 'Payment_Issue']:
        score += 2

    # Mentions of large money deduction (₹1000+)
    amounts = re.findall(r'₹?\s*(\d{4,6})', text)
    if amounts:
        score += 2

    # Long delays (7+ days, 2 weeks, ek mahina)
    if any(d in text_lower for d in ['7 din', '10 din', '15 din', '1 hafta', '2 hafta', 'ek mahina']):
        score += 2

    # ------------------------------------------------------------------------
    # 2. LOW URGENCY SIGNALS (-3 points each)
    # ------------------------------------------------------------------------
    low_keywords = [
        'usually kitne', 'kaise check karu', 'policy kya hai', 'process kya hai',
        'kaise karu', 'kya hota hai', 'option kahan hai', 'quick question', 'thanks', 'thank you'
    ]
    for kw in low_keywords:
        if kw in text_lower:
            score -= 3

    if category in ['App_Bug'] and not any(kw in text_lower for kw in ['crash', 'money', 'deduct', 'freeze']):
        score -= 2

    # ------------------------------------------------------------------------
    # 3. FINAL URGENCY LEVEL MAPPING
    # ------------------------------------------------------------------------
    if score >= 3:
        return 'High'
    elif score <= -1:
        return 'Low'
    else:
        return 'Medium'


def relabel_5k_dataset():
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(project_root, "data", "raw", "hinglish_hard_ambiguous_dataset_5000.csv")
    output_path = os.path.join(project_root, "data", "raw", "hinglish_hard_ambiguous_dataset_5000_relabeled.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input dataset at {input_path}")
        
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    print("Relabeling urgency column using semantic rule engine...")
    df['urgency'] = df.apply(lambda row: compute_semantic_urgency(row['text'], row['category']), axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"Saved relabeled dataset to: {output_path}")
    
    print("\nNew Urgency Distribution:")
    print(df['urgency'].value_counts())
    return output_path

if __name__ == "__main__":
    relabel_5k_dataset()
