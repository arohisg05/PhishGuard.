"""
PhishGuard - Experimental Domain-Aware Hybrid Inference Engine
--------------------------------------------------------------
A conservative, reputation-aware hybrid wrapper around Model 4 (CharNgramEngine):
Combines character n-gram subword lexical classification with deterministic,
Public Suffix List (PSL)-aware domain reputation from a versioned Tranco Top-100k snapshot.

Key Principles:
1. Zero Dependency: Pure Python 3.10+ standard library (math, json, os, re, urllib.parse).
2. Anti-Evasion Guarantee: Multi-tenant cloud platforms (pages.dev, firebaseapp.com, etc.),
   open file gateways (ipfs.io, gofile.io), link shorteners (bitly.ws), and unranked domains
   NEVER inherit parent domain reputation.
3. Conservative Risk Modulation: Legitimate popular domains receive a bounded log-odds
   discount, suppressing keyword seduction on authentic services without granting immunity
   to aggressive credential harvesters or compromised pages.
4. Explanatory Transparency: Provides full token attribution, domain rank, tenant isolation
   status, and log-odds adjustment for every analyzed URL.

Target Class Mapping:
- Class 0 = Phishing
- Class 1 = Legitimate
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from src.char_ngram_engine import CharNgramEngine

# Multi-tenant cloud hosting platforms, PaaS, dynamic site builders, and customer-facing infrastructure
# Subdomains on these platforms represent independent user tenants
MULTI_TENANT_SUFFIXES = {
    # Static & Jamstack / Serverless PaaS
    "pages.dev",
    "vercel.app",
    "netlify.app",
    "github.io",
    "gitlab.io",
    "firebaseapp.com",
    "web.app",
    "workers.dev",
    "render.com",
    "onrender.com",
    "surge.sh",
    "fly.dev",
    "railway.app",
    "supabase.co",
    "glitch.me",
    "replit.app",
    "repl.co",
    "wasmer.app",
    "lovable.app",
    "bolt.host",
    "zapier.app",

    # Website Builders & CMS Platforms
    "godaddysites.com",
    "squarespace.com",
    "wixsite.com",
    "wixstudio.com",
    "weebly.com",
    "weeblysite.com",
    "wordpress.com",
    "wpcomstaging.com",
    "site123.me",
    "strikingly.com",
    "jimdosite.com",
    "carrd.co",
    "myportfolio.com",
    "webflow.io",
    "square.site",
    "pantheonsite.io",
    "webcindario.com",
    "gitbook.io",
    "ghost.io",
    "myshopify.com",
    "blogspot.com",
    "000webhostapp.com",
    "framer.app",
    "framer.website",
    "business.site",
    "beacons.ai",
    "mez.ink",

    # Hosting & Managed Previews
    "cloudwaysapps.com",
    "oraclecloud.com",
    "myhuaweicloud.com",
    "cprapid.com",
    "plesk.page",
    "tw1.ru",
    "synology.me",
    "duckdns.org",
    "ngrok-free.app",
    "ngrok.io",
    "loca.lt",
    "herokuapp.com",

    # Public Cloud Storage & Gateways
    "pinata.cloud",
    "azurewebsites.net",
    "s3.amazonaws.com",
    "storage.googleapis.com",
    "blob.core.windows.net",
    "backblazeb2.com",
    "dweb.link",
    "cloudfunctions.net",
    "appspot.com",
    "cloudfront.net",
    "execute-api.amazonaws.com",
    "r2.dev",
    "b-cdn.net",
}

# Shared content platforms, public redirectors, URL shorteners, dynamic QR codes, and decentralized gateways
# Domains that allow arbitrary user uploads or redirects; must NEVER grant reputation credit
SHARED_REDIRECT_PLATFORMS = {
    "ipfs.io",
    "gofile.io",
    "bitly.ws",
    "link.me",
    "bio.link",
    "mandrillapp.com",
    "tinyurl.com",
    "bit.ly",
    "is.gd",
    "cutt.ly",
    "linktr.ee",
    "mega.nz",
    "pastebin.com",
    "raw.githubusercontent.com",
    "qrco.de",
    "ead.me",
    "wl.co",
    "scanned.page",
    "flowcode.com",
    "tr.ee",
}

# Recognized two-level country-code TLDs (PSL second-level suffixes)
TWO_LEVEL_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "net.uk", "ltd.uk", "plc.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
    "co.jp", "ne.jp", "ac.jp", "go.jp", "or.jp",
    "co.kr", "ne.kr", "go.kr", "ac.kr",
    "com.br", "net.br", "org.br", "gov.br", "edu.br",
    "co.in", "net.in", "org.in", "gov.in", "ac.in", "edu.in", "res.in",
    "com.mx", "gob.mx", "edu.mx", "org.mx",
    "com.sg", "gov.sg", "edu.sg", "org.sg",
    "co.za", "net.za", "org.za", "gov.za", "ac.za",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "gc.ca", "qc.ca", "on.ca", "bc.ca",
    "com.ar", "gob.ar", "edu.ar",
    "com.tr", "gov.tr", "edu.tr", "org.tr",
    "com.hk", "gov.hk", "edu.hk", "org.hk",
    "com.tw", "gov.tw", "edu.tw", "org.tw",
    "com.co", "gov.co", "edu.co",
    "com.pe", "gob.pe", "edu.pe",
    "com.ua", "gov.ua", "edu.ua",
    "com.my", "gov.my", "edu.my",
    "co.id", "go.id", "ac.id", "or.id",
    "com.vn", "gov.vn", "edu.vn",
}


class DomainAwareHybridEngine:
    """Experimental domain-aware wrapper around CharNgramEngine.

    Combines character n-gram scoring with PSL-aware Tranco domain reputation.
    """

    def __init__(
        self,
        char_model_path: str,
        tranco_snapshot_path: str,
        alpha: float = 1.3,
        threshold: float = 0.55,
        enable_http_bare_discount: bool = True,
    ):
        """Initialize the hybrid engine.

        Args:
            char_model_path: Path to serialized Model 4 JSON artifact.
            tranco_snapshot_path: Path to serialized Tranco Top-100k JSON dictionary.
            alpha: Log-odds attenuation scaling factor for top-ranked domains (default: 1.0).
            threshold: Classification decision threshold (default: 0.55).
            enable_http_bare_discount: Whether to damp unencrypted HTTP penalty on bare domains.
        """
        self.char_engine = CharNgramEngine(char_model_path)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.enable_http_bare_discount = bool(enable_http_bare_discount)

        if not os.path.isfile(tranco_snapshot_path):
            raise FileNotFoundError(f"Tranco snapshot not found at: {tranco_snapshot_path}")

        with open(tranco_snapshot_path, "r", encoding="utf-8") as f:
            self.tranco: Dict[str, int] = json.load(f)

    def parse_url_host(self, raw_url: str) -> Dict[str, Any]:
        """Parse and canonicalize URL hostname with PSL and multi-tenant awareness."""
        u_lower = raw_url.strip().lower()
        parsed = urlsplit(u_lower if "://" in u_lower else "http://" + u_lower)
        scheme = parsed.scheme
        path = parsed.path
        query = parsed.query
        raw_host = (parsed.hostname or "").strip(".")

        # Strip port if present
        host = raw_host.split(":")[0] if ":" in raw_host else raw_host

        # Check for IP literal
        is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) or "[" in host)

        # Strip leading www.
        if host.startswith("www."):
            host = host[4:]

        # 1. Check if host matches shared redirect / open gateway platforms
        if host in SHARED_REDIRECT_PLATFORMS or any(host.endswith("." + sp) for sp in SHARED_REDIRECT_PLATFORMS):
            return {
                "raw_host": raw_host,
                "canonical_host": host,
                "registered_domain": host,
                "scheme": scheme,
                "path": path,
                "query": query,
                "is_ip": is_ip,
                "is_multi_tenant": True,
                "is_shared_platform": True,
                "tranco_rank": None,
                "reputation_applicable": False,
            }

        # 2. Check if host is on a multi-tenant cloud hosting platform
        # Check AWS / Cloud infrastructure customer subdomains
        cloud_infras = (
            ".amazonaws.com",
            ".googleusercontent.com",
            ".cloudfunctions.net",
            ".appspot.com",
            ".azurewebsites.net",
            ".cloudapp.azure.com",
            ".cloudapp.net",
            ".blob.core.windows.net",
            ".azureedge.net",
            ".cloudfront.net",
            ".firebaseapp.com",
            ".web.app",
            ".pages.dev",
            ".workers.dev",
            ".vercel.app",
            ".netlify.app",
            ".github.io",
            ".gitlab.io",
            ".wifi.spectrum.com",
        )
        for infra in cloud_infras:
            if host.endswith(infra) and host != infra[1:]:
                return {
                    "raw_host": raw_host,
                    "canonical_host": host,
                    "registered_domain": host,
                    "scheme": scheme,
                    "path": path,
                    "query": query,
                    "is_ip": is_ip,
                    "is_multi_tenant": True,
                    "is_shared_platform": False,
                    "tranco_rank": None,
                    "reputation_applicable": False,
                }

        for suffix in MULTI_TENANT_SUFFIXES:
            if host.endswith("." + suffix):
                # Tenant is <subdomain>.<suffix> (e.g. evil-login.pages.dev or started-iotrexor.squarespace.com)
                subdomain_part = host[: -len(suffix) - 1]
                tenant_label = subdomain_part.split(".")[-1]
                # Corporate apex / www subdomains (e.g. www.squarespace.com) are not customer tenants
                if tenant_label not in ("www", ""):
                    tenant_boundary = f"{tenant_label}.{suffix}"
                    return {
                        "raw_host": raw_host,
                        "canonical_host": host,
                        "registered_domain": tenant_boundary,
                        "scheme": scheme,
                        "path": path,
                        "query": query,
                        "is_ip": is_ip,
                        "is_multi_tenant": True,
                        "is_shared_platform": False,
                        "tranco_rank": None,  # Tenant subdomain is unranked
                        "reputation_applicable": False,
                    }

        # 3. Handle IP addresses
        if is_ip:
            return {
                "raw_host": raw_host,
                "canonical_host": host,
                "registered_domain": host,
                "scheme": scheme,
                "path": path,
                "query": query,
                "is_ip": True,
                "is_multi_tenant": False,
                "is_shared_platform": False,
                "tranco_rank": None,
                "reputation_applicable": False,
            }

        # 4. Standard PSL-aware registered domain extraction
        parts = host.split(".")
        if len(parts) <= 2:
            reg_dom = host
        else:
            two_part = ".".join(parts[-2:])
            if two_part in TWO_LEVEL_TLDS and len(parts) >= 3:
                reg_dom = ".".join(parts[-3:])
            else:
                reg_dom = ".".join(parts[-2:])

        rank = self.tranco.get(reg_dom, None)
        rep_applicable = rank is not None and rank <= 100000

        return {
            "raw_host": raw_host,
            "canonical_host": host,
            "registered_domain": reg_dom,
            "scheme": scheme,
            "path": path,
            "query": query,
            "is_ip": False,
            "is_multi_tenant": False,
            "is_shared_platform": False,
            "tranco_rank": rank,
            "reputation_applicable": rep_applicable,
        }

    def compute_log_odds_adjustment(self, raw_url: str, raw_prob_phish: float) -> Tuple[float, Dict[str, Any]]:
        """Calculate the conservative reputation log-odds discount (Delta z)."""
        host_info = self.parse_url_host(raw_url)
        p_clamped = min(max(raw_prob_phish, 1e-6), 1.0 - 1e-6)
        z_raw = math.log(p_clamped / (1.0 - p_clamped))

        delta_z = 0.0
        adjustment_reasons: List[str] = []

        if host_info["reputation_applicable"]:
            rank = host_info["tranco_rank"]
            # Logarithmic reputation strength: Rank 1 -> 1.0, Rank 10 -> 0.8, Rank 100k -> 0.0
            rep_strength = max(0.0, 1.0 - (math.log10(rank) / 5.0))
            domain_discount = self.alpha * rep_strength
            delta_z += domain_discount
            adjustment_reasons.append(f"tranco_rank_{rank}_discount(-{round(domain_discount, 4)})")

            # Check unencrypted HTTP on bare domain (e.g. http://neverssl.com)
            path = host_info["path"]
            query = host_info["query"]
            is_bare = (not path or path == "/") and not query
            if self.enable_http_bare_discount and host_info["scheme"] == "http" and is_bare:
                delta_z += 1.1
                adjustment_reasons.append("http_bare_domain_scheme_damping(-1.10)")

            # Anti-Evasion Safety Gate:
            # If the raw log-odds is extremely high (z >= 2.2, p >= 0.90) indicating aggressive lure keywords,
            # restrict maximum reputation discount to avoid shielding phishing on compromised top domains.
            if z_raw >= 2.2:
                max_discount = 0.3
                if delta_z > max_discount:
                    delta_z = max_discount
                    adjustment_reasons.append(f"anti_evasion_cap_applied(max_-{max_discount})")

        return delta_z, {
            "host_info": host_info,
            "raw_log_odds": round(z_raw, 4),
            "delta_z": round(delta_z, 4),
            "adjustment_reasons": adjustment_reasons,
        }

    def predict_proba_sample(self, raw_url: str) -> List[float]:
        """Predict hybrid class probabilities: [P(Phishing), P(Legitimate)]."""
        p_raw = self.char_engine.predict_proba_sample(raw_url)[0]
        delta_z, _ = self.compute_log_odds_adjustment(raw_url, p_raw)

        p_clamped = min(max(p_raw, 1e-6), 1.0 - 1e-6)
        z_raw = math.log(p_clamped / (1.0 - p_clamped))
        z_adj = z_raw - delta_z

        # Numeric bounds
        z_adj = max(min(z_adj, 20.0), -20.0)
        p_phish = 1.0 / (1.0 + math.exp(-z_adj))
        p_legit = 1.0 - p_phish
        return [round(p_phish, 5), round(p_legit, 5)]

    def predict_sample(self, raw_url: str) -> int:
        """Predict binary class: 0 = Phishing, 1 = Legitimate.
        Uses the hybrid decision threshold (default: 0.55).
        """
        probs = self.predict_proba_sample(raw_url)
        return 0 if probs[0] >= self.threshold else 1

    def explain_sample(self, raw_url: str, top_k: int = 5) -> Dict[str, Any]:
        """Provide end-to-end explanation with subwords and domain reputation telemetry."""
        char_exp = self.char_engine.explain_sample(raw_url, top_k=top_k)
        p_raw = char_exp["prob_phishing"]
        delta_z, rep_telemetry = self.compute_log_odds_adjustment(raw_url, p_raw)

        probs_adj = self.predict_proba_sample(raw_url)
        pred_adj = self.predict_sample(raw_url)

        return {
            "url": raw_url,
            "final_prediction": "Phishing" if pred_adj == 0 else "Legitimate",
            "hybrid_prob_phishing": probs_adj[0],
            "hybrid_prob_legitimate": probs_adj[1],
            "raw_model4_prob_phishing": p_raw,
            "raw_model4_prediction": char_exp["prediction"],
            "decision_threshold": self.threshold,
            "reputation_telemetry": rep_telemetry,
            "top_phishing_subwords": char_exp["top_phishing_subwords"],
            "top_legitimate_subwords": char_exp["top_legitimate_subwords"],
        }
