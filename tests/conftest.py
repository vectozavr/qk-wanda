import torch


def pytest_sessionstart(session):
    # Tiny test models run faster without a large BLAS thread pool.
    torch.set_num_threads(1)
