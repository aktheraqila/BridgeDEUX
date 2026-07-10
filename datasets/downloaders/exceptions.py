"""
BridgeDEUX

Downloader Exception Hierarchy

Purpose
-------
Provide a structured exception hierarchy for the data ingestion
pipeline. Using dedicated exception types makes debugging easier,
improves log readability, and allows callers to recover from
specific failures.
"""

from __future__ import annotations


class DownloaderError(Exception):
    """
    Base class for every downloader-related exception.
    """

    pass


class DiscoveryError(DownloaderError):
    """
    Raised when dataset shards cannot be discovered.
    """

    pass


class DownloadError(DownloaderError):
    """
    Raised when a shard download fails.
    """

    pass


class VerificationError(DownloaderError):
    """
    Raised when a downloaded file fails integrity verification.
    """

    pass


class ManifestError(DownloaderError):
    """
    Raised when manifest generation or updates fail.
    """

    pass