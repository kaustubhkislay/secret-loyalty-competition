"""Ordinary CPU tests cannot open Internet connections or download models."""
import socket

import pytest


@pytest.fixture(autouse=True)
def offline_cpu_tests(request, monkeypatch):
    if request.node.get_closest_marker("model_training"):
        return
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def offline_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("CPU tests must not access the network")
        return original_connect(sock, address)

    def offline_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("CPU tests must not access the network")
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", offline_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", offline_connect_ex)
