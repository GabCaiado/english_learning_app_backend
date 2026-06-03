"""
generate_keys.py
----------------
Generates an RSA-4096 keypair for JWT signing (RS256).
Writes:
  keys/private.pem  – PKCS#8 private key (keep secret!)
  keys/public.pem   – Public key (safe to distribute)

Run once before starting the server for the first time:
  python scripts/generate_keys.py
"""

import os
import sys
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("ERROR: 'cryptography' package is required. Run: pip install cryptography")
    sys.exit(1)

KEYS_DIR = Path(__file__).resolve().parent.parent / "keys"


def generate_rsa_keypair(key_size: int = 4096) -> None:
    KEYS_DIR.mkdir(parents=True, exist_ok=True)

    private_path = KEYS_DIR / "private.pem"
    public_path = KEYS_DIR / "public.pem"

    if private_path.exists() and public_path.exists():
        print(f"Keys already exist in {KEYS_DIR}. Delete them manually to regenerate.")
        return

    print(f"Generating RSA-{key_size} keypair...")

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend(),
    )

    # Write private key (PKCS#8, PEM, unencrypted)
    with open(private_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    os.chmod(private_path, 0o600)  # owner-readable only

    # Write public key (PEM SubjectPublicKeyInfo)
    with open(public_path, "wb") as f:
        f.write(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    print(f"  Private key: {private_path}")
    print(f"  Public  key: {public_path}")
    print()
    print("For production deployments, store these as environment variables:")
    print("  JWT_PRIVATE_KEY='<contents of private.pem>'")
    print("  JWT_PUBLIC_KEY='<contents of public.pem>'")
    print()
    print("IMPORTANT: Never commit keys/ to version control!")


if __name__ == "__main__":
    generate_rsa_keypair()
