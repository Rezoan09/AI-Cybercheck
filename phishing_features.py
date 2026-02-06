import re
from collections import Counter

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
}

URGENT_KEYWORDS = {
    'urgent', 'immediately', 'action required', 'verify', 'suspend', 'limited time',
    'expire', 'confirm', 'alert', 'warning', 'attention', 'act now', 'click here',
    'urgent action', 'immediate action', 'asap', 'hurry', 'quick', 'fast', 'now'
}

def extract_phishing_features(text):
    """
    Extract numeric features from email text for phishing detection.
    
    Returns dict with 8 features matching the trained model.
    """
    if not text or not isinstance(text, str):
        return {
            'num_words': 0,
            'num_unique_words': 0,
            'num_stopwords': 0,
            'num_links': 0,
            'num_unique_domains': 0,
            'num_email_addresses': 0,
            'num_spelling_errors': 0,
            'num_urgent_keywords': 0
        }
    
    text_lower = text.lower()
    
    # Extract words
    words = re.findall(r'\b[a-z]+\b', text_lower)
    num_words = len(words)
    num_unique_words = len(set(words))
    
    # Count stopwords
    num_stopwords = sum(1 for word in words if word in STOPWORDS)
    
    # Extract links (http, https, www)
    links = re.findall(r'https?://[^\s]+|www\.[^\s]+', text_lower)
    num_links = len(links)
    
    # Extract unique domains
    domains = set()
    for link in links:
        domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/\s]+)', link)
        if domain_match:
            domains.add(domain_match.group(1))
    num_unique_domains = len(domains)
    
    # Extract email addresses
    emails = re.findall(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', text_lower)
    num_email_addresses = len(emails)
    
    # Estimate spelling errors (words not in common patterns, simplified approach)
    # Count words with unusual character patterns or excessive repetition
    num_spelling_errors = 0
    for word in words:
        if len(word) > 2:
            if re.search(r'(.)\1{2,}', word):
                num_spelling_errors += 1
            elif len(word) > 15:
                num_spelling_errors += 1
    
    # Count urgent keywords
    num_urgent_keywords = 0
    for keyword in URGENT_KEYWORDS:
        if keyword in text_lower:
            num_urgent_keywords += 1
    
    return {
        'num_words': num_words,
        'num_unique_words': num_unique_words,
        'num_stopwords': num_stopwords,
        'num_links': num_links,
        'num_unique_domains': num_unique_domains,
        'num_email_addresses': num_email_addresses,
        'num_spelling_errors': num_spelling_errors,
        'num_urgent_keywords': num_urgent_keywords
    }
