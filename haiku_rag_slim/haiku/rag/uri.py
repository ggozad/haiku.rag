import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def is_local_uri(uri: str) -> bool:
    """True for a ``file://`` URI or a bare filesystem path.

    ``urlparse("C:/docs/a.pdf")`` reports the drive letter as scheme ``c``.
    Require the separator after it so URI-like text such as ``x:content`` is
    not mistaken for a path.
    """
    scheme = urlparse(uri).scheme
    return scheme in ("", "file") or bool(_WINDOWS_ABSOLUTE_PATH.match(uri))


def uri_to_path(uri: str) -> Path:
    """Filesystem path for a ``file://`` URI or a bare path.

    ``file://`` URIs percent-encode special characters, and on Windows carry a
    leading slash before the drive (``file:///C:/docs``) that ``Path`` would
    keep. ``url2pathname`` handles both, per platform.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        host, path = parsed.netloc, parsed.path
        # file:////server/share spells a UNC path with an empty authority, the
        # host being the first path segment (RFC 8089 appendix E.3.2).
        if not host and path.startswith("//"):
            host, _, path = path[2:].partition("/")
        # url2pathname needs the host urlparse split off to build a UNC path,
        # but Python 3.14 rejects one handed to it on a non-Windows platform.
        # localhost denotes the current machine and is intentionally omitted.
        authority = f"//{host}" if host.lower() not in ("", "localhost") else ""
        return Path(f"{authority}{url2pathname('/' + path.lstrip('/'))}")
    if is_local_uri(uri):
        return Path(uri)
    raise ValueError(f"Not a local URI: {uri}")
