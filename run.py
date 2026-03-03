"""
Run the Travel Assistant application.
Usage: python run.py
"""
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import uvicorn
import datetime
import ipaddress

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
except Exception:
    x509 = None


def ensure_self_signed_cert(cert_path: str, key_path: str, common_name: str = "localhost"):
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return
    if x509 is None:
        raise RuntimeError("cryptography package is required to generate self-signed certificates.")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])

    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365 * 3))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(common_name),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    os.makedirs(os.path.dirname(cert_path), exist_ok=True)

    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Generated self-signed certificate: {cert_path} and key: {key_path}")

if __name__ == "__main__":
    # Ensure SSL cert and key exist (self-signed). Files are placed under ./ssl/
    project_root = os.path.dirname(__file__)
    ssl_dir = os.path.join(project_root, "ssl")
    cert_file = os.path.join(ssl_dir, "cert.pem")
    key_file = os.path.join(ssl_dir, "key.pem")

    try:
        ensure_self_signed_cert(cert_file, key_file)
    except RuntimeError as e:
        print(str(e))
        print("Install the 'cryptography' package (pip install cryptography) to enable cert generation.")
        raise

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        ssl_certfile=cert_file,
        ssl_keyfile=key_file,
    )
