"""
Hinglish Text Preprocessor (Enhanced v2)
Handles cleaning, normalization, and preprocessing of Hinglish (Hindi-English code-mixed) text.
Now includes urgency cue detection for better urgency classification.
"""
import re
import string
import pickle
import os


class HinglishPreprocessor:
    """
    Preprocessor for Hinglish (Hindi-English code-mixed) e-commerce complaint text.

    Handles:
    - Lowercasing
    - Spelling variant normalization (nahi/nai/nahee -> nahin)
    - URL, email, phone number replacement with tokens
    - Currency amount replacement with tokens
    - Urgency cue detection (CAPS, exclamation, threats, escalation)
    - Repeated letter normalization (urgenttt -> urgent)
    - Special character removal
    - Hindi + English stopword removal
    - Short word removal (single characters)
    """

    def __init__(self):
        # Common Hindi stopwords (Roman script)
        self.hindi_stopwords = {
            'hai', 'hain', 'tha', 'thi', 'the', 'ho', 'hun',
            'ka', 'ki', 'ke', 'ko', 'se', 'me', 'mein', 'pe', 'ne',
            'aur', 'ya', 'to', 'ye', 'wo', 'yeh', 'woh',
            'kya', 'kaise', 'kyun', 'kab', 'kaha', 'kahan',
            'main', 'mera', 'meri', 'mere',
            'aap', 'tum', 'tu', 'apna', 'tera',
            'koi', 'kuch', 'sab', 'yah', 'vah',
            'abhi', 'phir', 'bhi', 'hi',
            'par', 'lekin', 'magar', 'jaise',
            'do', 'de', 'diya', 'hua', 'hui', 'huye',
            'ek', 'do', 'tin', 'char',
            'iski', 'iska', 'iske', 'uski', 'uska', 'uske',
            'jiski', 'jiska', 'jiske',
            'isi', 'isa', 'ise', 'usko', 'usse',
            'yahan', 'wahan', 'jahan', 'kahin',
            'ab', 'aj', 'kal', 'aaj',
            'ji', 'shri', 'sri',
        }

        # English stopwords (common ones)
        self.english_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
            'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
            'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
            'shouldn', 'wasn', 'weren', 'won', 'wouldn',
        }

        self.all_stopwords = self.hindi_stopwords | self.english_stopwords

        # Hinglish spelling normalization map
        self.spelling_map = {
            'nahi': 'nahin', 'nai': 'nahin', 'nahee': 'nahin', 'nahin': 'nahin',
            'bahut': 'bahut', 'bahot': 'bahut', 'bhot': 'bahut', 'bohot': 'bahut',
            'koi': 'koi', 'koee': 'koi',
            'kya': 'kya', 'kia': 'kya', 'kiya': 'kya',
            'hai': 'hai', 'he': 'hai',
            'se': 'se', 'sey': 'se', 'seh': 'se',
            'ye': 'yeh', 'yae': 'yeh',
            'wo': 'woh', 'vo': 'woh',
            'kal': 'kal', 'aj': 'aaj', 'aaj': 'aaj',
            'jaldi': 'jaldi', 'jldi': 'jaldi',
            'turant': 'turant', 'turanat': 'turant',
            'urgent': 'urgent', 'urgenttt': 'urgent', 'urgnt': 'urgent',
            'refund': 'refund', 'refnd': 'refund', 'refound': 'refund',
            'delivery': 'delivery', 'delivry': 'delivery', 'delievery': 'delivery',
            'delivery': 'delivery', 'delivr': 'delivery',
            'order': 'order', 'ordr': 'order', 'oder': 'order',
            'payment': 'payment', 'paymnt': 'payment', 'payement': 'payment',
            'return': 'return', 'retrn': 'return', 'reurn': 'return',
            'product': 'product', 'prodct': 'product', 'prduct': 'product',
            'account': 'account', 'acount': 'account', 'acconut': 'account',
            'issue': 'issue', 'isue': 'issue',
            'complaint': 'complaint', 'complint': 'complaint',
            'package': 'package', 'parcl': 'package', 'pkg': 'package', 'parcel': 'package',
            'track': 'track', 'trackng': 'track', 'trak': 'track',
            'address': 'address', 'adress': 'address', 'adres': 'address',
            'invoice': 'invoice', 'invioce': 'invoice', 'invocie': 'invoice',
            'number': 'number', 'nmber': 'number', 'numbr': 'number',
            'correct': 'correct', 'corect': 'correct',
            'today': 'today', 'toady': 'today',
            'urgent': 'urgent', 'urgnt': 'urgent',
            'please': 'please', 'plss': 'please', 'plz': 'please',
            'thanks': 'thanks', 'thnks': 'thanks',
            'okay': 'okay', 'oky': 'okay',
            'also': 'also', 'aloso': 'also',
            'problem': 'problem', 'problm': 'problem',
            'solution': 'solution', 'solotion': 'solution',
            'check': 'check', 'chek': 'check',
            'update': 'update', 'upadte': 'update',
            'response': 'response', 'resposne': 'response',
            'immediately': 'immediately', 'immeditely': 'immediately', 'immediatly': 'immediately',
            'important': 'important', 'importnat': 'important',
            'cancellation': 'cancellation', 'cancelation': 'cancellation',
            'replacement': 'replacement', 'replcement': 'replacement',
            'damaged': 'damaged', 'dmaged': 'damaged',
            'wrong': 'wrong', 'wrng': 'wrong',
            'missing': 'missing', 'msising': 'missing',
            'delivered': 'delivered', 'deliverd': 'delivered',
            'received': 'received', 'recieved': 'received', 'recieved': 'received',
            'customer': 'customer', 'custmer': 'customer',
            'support': 'support', 'suport': 'support',
            'agent': 'agent', 'aggent': 'agent',
        }

        # Urgency cue patterns
        self.threat_words = {
            'consumer court', 'legal', 'police', 'court', 'lawyer', 'advocate',
            'cyber crime', 'fraud', 'cheating', 'scam', 'file case', 'sue',
            'consumer forum', 'legal action', 'legal notice',
        }

        self.escalation_words = {
            'manager', 'senior', 'escalate', 'escalation', 'higher authority',
            'supervisor', 'nodal officer', 'grievance', 'ombudsman',
        }

        self.high_value_pattern = re.compile(r'(?:rs\.?|₹)\s*(\d[\d,]*)', re.IGNORECASE)

    def detect_urgency_cues(self, text):
        """
        Detect urgency cues in text and return tokens to add.

        Returns a string of urgency tokens to append to the preprocessed text.
        """
        cues = []
        original_text = text  # Keep original for detection

        # 1. ALL CAPS words (at least 2 chars, not common words)
        caps_words = re.findall(r'\b([A-Z]{2,})\b', original_text)
        meaningful_caps = [w for w in caps_words if len(w) >= 2 and w not in {'OK', 'GST', 'URL', 'SMS', 'UPI'}]
        if meaningful_caps:
            cues.append('URGENTCAPS')

        # 2. Exclamation marks intensity
        excl_count = original_text.count('!')
        if excl_count >= 3:
            cues.append('EXCLAMATIONHIGH')
        elif excl_count >= 1:
            cues.append('EXCLAMATION')

        # 3. Question marks (frustration indicator)
        quest_count = original_text.count('?')
        if quest_count >= 2:
            cues.append('MULTIPLEQUESTIONS')

        # 4. Threat words
        text_lower = original_text.lower()
        for threat in self.threat_words:
            if threat in text_lower:
                cues.append('THREAT')
                break

        # 5. Escalation words
        for esc in self.escalation_words:
            if esc in text_lower:
                cues.append('ESCALATION')
                break

        # 6. High-value amounts (₹5000+, ₹10000+)
        amount_matches = self.high_value_pattern.findall(original_text)
        for amt_str in amount_matches:
            try:
                amt = int(amt_str.replace(',', ''))
                if amt >= 10000:
                    cues.append('HIGHAMOUNT')
                    break
                elif amt >= 5000:
                    cues.append('MEDIUMAMOUNT')
                    break
            except ValueError:
                pass

        # 7. Urgency words in Hindi/English
        urgency_keywords = [
            'urgent', 'jaldi', 'turant', 'abhi', 'fauran', 'tatkal',
            'immediately', 'asap', 'emergency', 'critical',
        ]
        for kw in urgency_keywords:
            if kw in text_lower:
                cues.append('URGENTKEYWORD')
                break

        # 8. Time pressure words
        time_pressure = [
            'din', 'days', 'weeks', 'months', 'wait', 'intezaar', 'intejar',
            'pending', 'lambe', 'bahut din',
        ]
        for tp in time_pressure:
            if tp in text_lower:
                cues.append('TIMEPRESSURE')
                break

        return ' '.join(cues) if cues else ''

    def normalize_repeated_letters(self, text):
        """Normalize repeated letters: urgenttt -> urgent, plssss -> plss."""
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def normalize_spelling(self, text):
        """Normalize Hinglish spelling variants to canonical forms."""
        words = text.lower().split()
        normalized = [self.spelling_map.get(w, w) for w in words]
        return ' '.join(normalized)

    def clean_text(self, text):
        """Clean raw text: remove noise and replace special tokens."""
        if not isinstance(text, str):
            return "", ""

        # Detect urgency cues BEFORE cleaning
        urgency_tokens = self.detect_urgency_cues(text)

        text = text.lower()
        text = self.normalize_repeated_letters(text)
        text = self.normalize_spelling(text)

        # Replace URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' URLTOKEN ', text)
        # Replace email addresses
        text = re.sub(r'\S+@\S+', ' EMAILTOKEN ', text)
        # Replace phone numbers (10 digits)
        text = re.sub(r'\b\d{10}\b', ' PHONETOKEN ', text)
        # Replace currency amounts (₹ symbol followed by digits)
        text = re.sub(r'₹[\d,]+', ' AMOUNTTOKEN ', text)
        # Replace "rs." followed by digits
        text = re.sub(r'rs\.?\s*[\d,]+', ' AMOUNTTOKEN ', text)
        # Replace standalone numbers
        text = re.sub(r'\b\d+\b', ' NUMBERTOKEN ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text, urgency_tokens

    def remove_stopwords(self, text):
        """Remove Hindi and English stopwords."""
        words = text.split()
        filtered = [w for w in words if w not in self.all_stopwords and len(w) > 1]
        return ' '.join(filtered)

    def preprocess(self, text):
        """
        Full preprocessing pipeline:
        1. Detect urgency cues (before cleaning)
        2. Clean text (lowercase, normalize spelling, replace tokens, remove special chars)
        3. Remove stopwords
        4. Filter short words
        5. Append urgency tokens

        Returns: cleaned string with urgency tokens
        """
        text, urgency_tokens = self.clean_text(text)
        text = self.remove_stopwords(text)

        # Append urgency tokens to the end of the text
        if urgency_tokens:
            text = f"{text} {urgency_tokens}"

        return text

    def preprocess_batch(self, texts):
        """Preprocess a batch of texts. Returns list of cleaned texts."""
        return [self.preprocess(t) for t in texts]

    def save(self, filepath):
        """Save preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load preprocessor from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
