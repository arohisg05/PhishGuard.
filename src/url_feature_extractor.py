"""
PhishGuard URL Feature Extractor
--------------------------------
Extracts deterministic lexical and structural features directly from a raw URL
string without network requests, web scraping, or third-party dependencies.

Outputs exactly 19 numerical features aligned with the PhiUSIIL dataset schema
for offline URL-only classification.
"""

from __future__ import annotations

import ipaddress
import math
import re
from typing import Any, Dict
from urllib.parse import urlsplit

# Exact ordered feature list expected by PhishGuard URL models
URL_FEATURE_NAMES = (
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "LongURL",
)

# Regex pattern for standard percent-encoded characters (e.g. %20, %3D, %2F)
PERCENT_ENCODING_PATTERN = re.compile(r"%[0-9a-fA-F]{2}")

# Supported web schemes
SUPPORTED_SCHEMES = ("http", "https", "ftp")


def extract_url_features(raw_url: str) -> Dict[str, float]:
    """Extract 19 numerical features from a raw URL string.

    Args:
        raw_url: The URL string to inspect (with or without http/https scheme).

    Returns:
        Dict[str, float]: An ordered dictionary containing exactly 19 numerical
                          features.

    Raises:
        TypeError: If raw_url is not a string.
        ValueError: If raw_url is empty, malformed, or uses an unsupported scheme.
    """
    # -------------------------------------------------------------------------
    # 1. Input Validation
    # -------------------------------------------------------------------------
    if not isinstance(raw_url, str):
        raise TypeError(f"Expected URL as string, got {type(raw_url).__name__}")

    url = raw_url.strip()
    if not url:
        raise ValueError("URL string cannot be empty or contain only whitespace.")

    # Reject non-printable / control characters
    if any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("URL contains illegal unprintable or control characters.")

    # -------------------------------------------------------------------------
    # 2. Scheme & Hostname Parsing
    # -------------------------------------------------------------------------
    # Check for an explicit URI scheme (e.g. 'https://...', 'ftp://...', 'javascript:...', 'data:...')
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", url)
    if scheme_match:
        potential_scheme = scheme_match.group(1).lower()
        after_colon = url[scheme_match.end():]
        # Distinguish genuine schemes (http://, mailto:..., javascript:...) from port numbers (host.com:8080)
        if after_colon.startswith("//") or not re.match(r"^[0-9]+(/|$)", after_colon):
            if potential_scheme not in SUPPORTED_SCHEMES:
                raise ValueError(
                    f"Unsupported URL scheme '{potential_scheme}:'. Supported schemes: {SUPPORTED_SCHEMES}"
                )
            scheme = potential_scheme
            is_https = 1.0 if scheme == "https" else 0.0
            parse_target = url
        else:
            # Matches host:port pattern (e.g. 'example.com:8080')
            scheme = ""
            is_https = 0.0
            parse_target = "http://" + url
    else:
        # Scheme-less URL (e.g. 'www.example.com/path')
        scheme = ""
        is_https = 0.0
        parse_target = "http://" + url

    try:
        parsed = urlsplit(parse_target)
    except Exception as exc:
        raise ValueError(f"Malformed URL could not be parsed: {exc}") from exc

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(
            f"Malformed URL: unable to extract a valid host/domain name from '{raw_url}'."
        )

    # -------------------------------------------------------------------------
    # 3. Domain, IP, and TLD Calculations
    # -------------------------------------------------------------------------
    domain_length = float(len(hostname))

    # Check if host is an IPv4 or IPv6 address
    try:
        ipaddress.ip_address(hostname)
        is_domain_ip = 1.0
    except ValueError:
        is_domain_ip = 0.0

    # TLD and Subdomain extraction
    # In PhiUSIIL, TLD is taken as the final dot-separated segment, and subdomains
    # are counted as max(0, number_of_segments - 2), e.g.:
    # - 'example.com'        -> parts: ['example', 'com']         -> 0 subdomains, TLD: 'com' (len 3)
    # - 'www.example.com'    -> parts: ['www', 'example', 'com']  -> 1 subdomain,  TLD: 'com' (len 3)
    # - '192.168.1.1' (IP)   -> TLDLength: 0, NoOfSubDomain: 0
    if is_domain_ip == 1.0:
        tld_length = 0.0
        no_of_subdomain = 0.0
    else:
        parts = hostname.split(".")
        if len(parts) > 1:
            tld = parts[-1]
            tld_length = float(len(tld))
            no_of_subdomain = float(max(0, len(parts) - 2))
        else:
            tld_length = 0.0
            no_of_subdomain = 0.0

    # -------------------------------------------------------------------------
    # 4. Character Composition & Counts
    # -------------------------------------------------------------------------
    url_length = float(len(url))

    # Letters: [a-zA-Z]
    no_of_letters = float(sum(1 for c in url if c.isalpha()))
    letter_ratio = (no_of_letters / url_length) if url_length > 0 else 0.0

    # Digits: [0-9]
    no_of_digits = float(sum(1 for c in url if c.isdigit()))
    digit_ratio = (no_of_digits / url_length) if url_length > 0 else 0.0

    # Specific delimiters
    no_of_equals = float(url.count("="))
    no_of_qmark = float(url.count("?"))
    no_of_ampersand = float(url.count("&"))

    # Other special characters: non-alphanumeric characters excluding '=', '?', and '&'
    # e.g., ':', '/', '.', '-', '_', '%', etc.
    no_of_other_special = float(
        sum(1 for c in url if not c.isalnum() and c not in ("=", "?", "&"))
    )
    special_char_ratio = (no_of_other_special / url_length) if url_length > 0 else 0.0

    # -------------------------------------------------------------------------
    # 5. Obfuscation & Feature Engineering
    # -------------------------------------------------------------------------
    # Detect percent-encoded hex sequences (%xx)
    obfuscated_matches = PERCENT_ENCODING_PATTERN.findall(url)
    if obfuscated_matches:
        has_obfuscation = 1.0
        # In PhiUSIIL, each %xx token occupies 3 characters
        no_of_obfuscated_char = float(len(obfuscated_matches) * 3)
    else:
        has_obfuscation = 0.0
        no_of_obfuscated_char = 0.0

    obfuscation_ratio = (no_of_obfuscated_char / url_length) if url_length > 0 else 0.0

    # Day 5 Engineered feature: LongURL = 1 if URLLength > 100, else 0
    long_url = 1.0 if url_length > 100.0 else 0.0

    # -------------------------------------------------------------------------
    # 6. Assemble Exact Ordered Dictionary
    # -------------------------------------------------------------------------
    features: Dict[str, float] = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": round(letter_ratio, 6),
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": round(digit_ratio, 6),
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special,
        "SpacialCharRatioInURL": round(special_char_ratio, 6),
        "IsHTTPS": is_https,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated_char,
        "ObfuscationRatio": round(obfuscation_ratio, 6),
        "LongURL": long_url,
    }

    return features


# =============================================================================
# SELF-TEST SUITE
# =============================================================================
def run_self_test() -> bool:
    """Run verification self-tests on diverse URLs and edge cases."""
    print("=" * 70)
    print("RUNNING URL FEATURE EXTRACTOR SELF-TEST")
    print("=" * 70)

    test_cases = [
        {
            "name": "Standard HTTPS with Query Params",
            "url": "https://www.example.com/login?user=test&session=123",
            "expected_checks": {
                "IsHTTPS": 1.0,
                "IsDomainIP": 0.0,
                "NoOfSubDomain": 1.0,  # 'www'
                "TLDLength": 3.0,      # 'com'
                "NoOfEqualsInURL": 2.0,
                "NoOfQMarkInURL": 1.0,
                "NoOfAmpersandInURL": 1.0,
                "LongURL": 0.0,
            },
        },
        {
            "name": "Raw IP Address URL",
            "url": "http://192.168.1.1:8080/admin",
            "expected_checks": {
                "IsHTTPS": 0.0,
                "IsDomainIP": 1.0,
                "TLDLength": 0.0,
                "NoOfSubDomain": 0.0,
            },
        },
        {
            "name": "Scheme-less URL with Percent Obfuscation",
            "url": "example.org/search%20results%3Fid%3D1",
            "expected_checks": {
                "IsHTTPS": 0.0,
                "IsDomainIP": 0.0,
                "TLDLength": 3.0,
                "HasObfuscation": 1.0,
                "NoOfObfuscatedChar": 9.0,  # 3 matches * 3 chars = 9
            },
        },
        {
            "name": "Unusually Long Phishing-Style URL (>100 chars)",
            "url": "https://secure-login.bank-verification.com/auth/portal/update/verification?token=abcdef1234567890abcdef1234567890&redirect=true",
            "expected_checks": {
                "IsHTTPS": 1.0,
                "LongURL": 1.0,
                "NoOfSubDomain": 1.0,
            },
        },
    ]

    all_passed = True

    # 1. Test Valid URL Outputs & Keys
    for tc in test_cases:
        name = tc["name"]
        url = tc["url"]
        try:
            feats = extract_url_features(url)
            # Verify exact 19 keys
            keys = tuple(feats.keys())
            if keys != URL_FEATURE_NAMES:
                print(f"[FAIL] {name}: Returned keys do not match expected 19 keys in order.")
                all_passed = False
                continue

            # Verify values are numeric and finite
            for k, v in feats.items():
                if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    print(f"[FAIL] {name}: Feature '{k}' has invalid value {v}")
                    all_passed = False

            # Verify expected values
            for exp_k, exp_v in tc["expected_checks"].items():
                if feats[exp_k] != exp_v:
                    print(f"[FAIL] {name}: Feature '{exp_k}' expected {exp_v}, got {feats[exp_k]}")
                    all_passed = False

            print(f"[PASS] {name} -> 19 features extracted successfully.")

        except Exception as exc:
            print(f"[FAIL] {name} raised unexpected exception: {exc}")
            all_passed = False

    # 2. Test Invalid Inputs (Must Raise ValueError/TypeError)
    invalid_cases = [
        ("Empty String", ""),
        ("Whitespace Only", "     "),
        ("Non-String (None)", None),
        ("Non-String (Integer)", 12345),
        ("Unsupported Scheme (javascript:)", "javascript:alert(1)"),
        ("Unsupported Data Scheme (data:)", "data:text/html,<b>hi</b>"),
        ("Control Character", "https://example.com/\x00bad"),
    ]

    print("\nTesting Invalid Inputs Rejection:")
    for name, inv_input in invalid_cases:
        try:
            extract_url_features(inv_input)  # type: ignore
            print(f"[FAIL] Expected exception for {name}, but call succeeded.")
            all_passed = False
        except (ValueError, TypeError) as exc:
            print(f"[PASS] Correctly rejected {name} with: {type(exc).__name__} ('{exc}')")

    print("-" * 70)
    if all_passed:
        print("ALL SELF-TESTS PASSED SUCCESSFULLY (100%).")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    run_self_test()
