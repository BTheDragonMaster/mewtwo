def ts_to_te(ts: float) -> float:
    """
    Return terminator efficiency from terminator strength

    Parameters
    ----------
    ts: float, termination strength, as defined by Chen et al., 2013

    Returns
    -------
    te: float, termination efficiency
    """

    return 1 - 1 / ts
