"""
Digital Signature Extractor
============================
Extracts digital signature metadata from native (non-scanned) PDF files.

Uses PyMuPDF to locate /Sig form fields, then the ``cryptography`` library to
parse the embedded PKCS#7 / CMS SignedData structure and extract signer details,
validity period, and signature status.

Each detected signature is returned as a dict:
  {
    "signer_name":   str | None,   # Common Name from signer certificate
    "organization":  str | None,   # O field from signer certificate
    "email":         str | None,   # email from signer certificate
    "issuer":        str | None,   # CA Common Name
    "valid_from":    str | None,   # ISO-8601, certificate notBefore
    "valid_until":   str | None,   # ISO-8601, certificate notAfter
    "signing_time":  str | None,   # ISO-8601 from CMS signingTime attribute
    "field_name":    str,          # PDF form field name (e.g. "Signature1")
    "page":          int,          # 1-based page number of the field
    "reason":        str | None,   # /Reason entry in the sig dict
    "location":      str | None,   # /Location entry in the sig dict
  }

Returns an empty list for:
- Non-PDF files
- PDFs with no /Sig form fields
- Any parse error (logged as WARNING, never raises)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _attr_value(name_attrs, oid_dotted: str) -> str | None:
    """Extract a specific OID value from an x509.Name attribute list."""
    try:
        from cryptography.x509.oid import NameOID
        from cryptography import x509
        # Build a map of dotted_string → value
        for attr in name_attrs:
            if attr.oid.dotted_string == oid_dotted:
                return str(attr.value)
    except Exception:
        pass
    return None


def _parse_pkcs7(raw: bytes) -> dict[str, Any]:
    """
    Parse PKCS#7 / CMS SignedData from raw DER bytes.
    Returns a dict with keys: signer_name, organization, email,
    issuer, valid_from, valid_until, signing_time.
    """
    from cryptography.hazmat.primitives.serialization.pkcs7 import (
        load_der_pkcs7_certificates,
    )
    from cryptography.x509.oid import ExtensionOID

    result: dict[str, Any] = {
        "signer_name": None,
        "organization": None,
        "email": None,
        "issuer": None,
        "valid_from": None,
        "valid_until": None,
        "signing_time": None,
    }

    # ── Extract certificates from the PKCS#7 blob ──────────────────
    try:
        certs = load_der_pkcs7_certificates(raw)
    except Exception:
        # Might be PEM-wrapped or contain only the signature without certs
        return result

    if not certs:
        return result

    # The end-entity cert is typically the first one (leaf)
    leaf = certs[0]
    subject = leaf.subject
    issuer_name = leaf.issuer

    # Common OIDs as dotted strings
    OID_CN    = "2.5.4.3"
    OID_O     = "2.5.4.10"
    OID_EMAIL = "1.2.840.113549.1.9.1"

    result["signer_name"]  = _attr_value(subject, OID_CN)
    result["organization"] = _attr_value(subject, OID_O)
    result["email"]        = _attr_value(subject, OID_EMAIL)
    result["issuer"]       = _attr_value(issuer_name, OID_CN)
    result["valid_from"]   = _iso(leaf.not_valid_before_utc if hasattr(leaf, "not_valid_before_utc") else leaf.not_valid_before)
    result["valid_until"]  = _iso(leaf.not_valid_after_utc if hasattr(leaf, "not_valid_after_utc") else leaf.not_valid_after)

    # Try SAN for email if not in subject
    if not result["email"]:
        try:
            san = leaf.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            from cryptography.x509 import RFC822Name
            emails = san.value.get_values_for_type(RFC822Name)
            if emails:
                result["email"] = emails[0]
        except Exception:
            pass

    return result


def _extract_signing_time_from_raw(raw: bytes) -> str | None:
    """
    Try to extract the signingTime CMS attribute from raw PKCS#7 bytes.
    Uses asn1crypto for a lightweight ASN.1 parse — falls back gracefully.
    """
    try:
        from asn1crypto import cms as asn1_cms
        content_info = asn1_cms.ContentInfo.load(raw)
        signed_data = content_info["content"].parsed
        for signer_info in signed_data["signer_infos"]:
            for attr in signer_info["signed_attrs"]:
                if attr["type"].native == "signing_time":
                    val = attr["values"][0].native
                    if isinstance(val, datetime):
                        return _iso(val)
                    return str(val)
    except Exception:
        pass
    return None


def extract_digital_signatures(file_path: str) -> list[dict[str, Any]]:
    """
    Synchronous function — run via asyncio.to_thread in async contexts.

    Parses a PDF and returns a list of digital signature metadata dicts.
    Returns [] for non-PDFs, unsigned PDFs, or any error.
    """
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return []

    signatures: list[dict[str, Any]] = []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("[digital_sig] PyMuPDF not available — skipping signature extraction")
        return []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.warning(f"[digital_sig] Failed to open PDF {path.name}: {e}")
        return []

    try:
        # Iterate all pages looking for signature widgets
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for widget in page.widgets() or []:
                if widget.field_type != fitz.PDF_WIDGET_TYPE_SIGNATURE:
                    continue

                field_name = widget.field_name or f"sig_{page_idx + 1}"

                sig_entry: dict[str, Any] = {
                    "field_name":  field_name,
                    "page":        page_idx + 1,
                    "signer_name": None,
                    "organization": None,
                    "email":       None,
                    "issuer":      None,
                    "valid_from":  None,
                    "valid_until": None,
                    "signing_time": None,
                    "reason":      None,
                    "location":    None,
                }

                # Try to get reason / location from the signature dict
                try:
                    xref = widget.xref
                    sig_dict = doc.xref_get_key(xref, "V")
                    if sig_dict and sig_dict[0] == "xref":
                        inner_xref = int(sig_dict[1].strip().split()[0])
                        reason_raw   = doc.xref_get_key(inner_xref, "Reason")
                        location_raw = doc.xref_get_key(inner_xref, "Location")
                        if reason_raw and reason_raw[0] not in ("null", ""):
                            sig_entry["reason"] = reason_raw[1].strip("()")
                        if location_raw and location_raw[0] not in ("null", ""):
                            sig_entry["location"] = location_raw[1].strip("()")
                except Exception:
                    pass

                # Extract PKCS#7 bytes from the /Contents entry
                try:
                    raw_bytes: bytes | None = None
                    xref = widget.xref
                    contents_val = doc.xref_get_key(xref, "V")
                    if contents_val and contents_val[0] == "xref":
                        inner_xref = int(contents_val[1].strip().split()[0])
                        contents_raw = doc.xref_get_key(inner_xref, "Contents")
                        if contents_raw and contents_raw[0] == "string":
                            # PyMuPDF returns hex-encoded or raw bytes as str
                            raw_str = contents_raw[1]
                            if raw_str.startswith("<") and raw_str.endswith(">"):
                                raw_bytes = bytes.fromhex(raw_str[1:-1])
                            else:
                                raw_bytes = raw_str.encode("latin-1")

                    if raw_bytes:
                        cert_info = _parse_pkcs7(raw_bytes)
                        sig_entry.update(cert_info)
                        if not sig_entry["signing_time"]:
                            sig_entry["signing_time"] = _extract_signing_time_from_raw(raw_bytes)

                except Exception as e:
                    logger.debug(f"[digital_sig] Could not parse PKCS7 for {field_name}: {e}")

                signatures.append(sig_entry)

    except Exception as e:
        logger.warning(f"[digital_sig] Error extracting signatures from {path.name}: {e}")
    finally:
        doc.close()

    if signatures:
        logger.info(
            f"[digital_sig] {path.name}: found {len(signatures)} signature(s): "
            + ", ".join(s["field_name"] for s in signatures)
        )
    return signatures
