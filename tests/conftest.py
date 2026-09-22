# -*- coding: utf-8 -*-
"""pytest 共享 fixture"""
import logging
import pytest

@pytest.fixture(scope="session")
def logger():
    """返回测试 logger"""
    return logging.getLogger("pytest")
