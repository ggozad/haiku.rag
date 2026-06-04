import socket


def reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if a TCP connection to host:port succeeds within timeout. Used by
    integration tests to skip when their backing service isn't running."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
