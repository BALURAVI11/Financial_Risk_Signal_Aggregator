def format_inr(amount) -> str:
    """Formats a number using the Indian digit-grouping convention, e.g. 4774935.7 -> ₹47,74,935.70."""
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    whole = int(amount)
    frac = f"{amount - whole:.2f}".split(".")[1]
    s = str(whole)
    if len(s) <= 3:
        grouped = s
    else:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}₹{grouped}.{frac}"
