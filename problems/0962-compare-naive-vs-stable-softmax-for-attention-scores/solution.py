import math


def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""

    # ---------- naive softmax ----------
    try:
        exp_naive = [math.exp(x) for x in scores]
        sum_naive = sum(exp_naive)

        if sum_naive == 0:
            naive = [float("nan")] * len(scores)
        else:
            naive = [
                round(x / sum_naive, 6)
                for x in exp_naive
            ]

    except OverflowError:
        # 只要有一个 exp 溢出，
        # 这道题要求整个 naive softmax 都视为 nan
        naive = [float("nan")] * len(scores)

    # ---------- stable softmax ----------
    max_v = max(scores)

    exp_stable = [
        math.exp(x - max_v)
        for x in scores
    ]

    sum_stable = sum(exp_stable)

    stable = [
        round(x / sum_stable, 6)
        for x in exp_stable
    ]

    # ---------- max absolute difference ----------
    if any(math.isnan(x) for x in naive):
        max_abs_diff = float("nan")
    else:
        max_abs_diff = round(
            max(
                abs(a - b)
                for a, b in zip(naive, stable)
            ),
            6
        )

    return {
        "naive": naive,
        "stable": stable,
        "max_abs_diff": max_abs_diff
    }