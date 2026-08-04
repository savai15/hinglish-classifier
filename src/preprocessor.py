"""
Hinglish Text Preprocessor
Handles cleaning, normalization, and preprocessing of Hinglish (Hindi-English code-mixed) text.
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

    def normalize_spelling(self, text):
        """Normalize Hinglish spelling variants to canonical forms."""
        words = text.lower().split()
        normalized = [self.spelling_map.get(w, w) for w in words]
        return ' '.join(normalized)

    def clean_text(self, text):
        """Clean raw text: remove noise and replace special tokens."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
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

        return text

    def remove_stopwords(self, text):
        """Remove Hindi and English stopwords."""
        words = text.split()
        filtered = [w for w in words if w not in self.all_stopwords and len(w) > 1]
        return ' '.join(filtered)

    def preprocess(self, text):
        """
        Full preprocessing pipeline:
        1. Clean text (lowercase, normalize spelling, replace tokens, remove special chars)
        2. Remove stopwords
        3. Filter short words

        Returns: cleaned string
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

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
