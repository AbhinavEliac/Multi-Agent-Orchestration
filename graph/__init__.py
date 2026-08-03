def __getattr__(name):
    if name == "graph":
        from .graph import graph

        return graph

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

