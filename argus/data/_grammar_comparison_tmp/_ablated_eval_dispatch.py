
def evaluate_ablated(op, vals):
    """A counterfactual Window.evaluate that dispatches by building and eval()-ing a Python
    expression from the op name — the shortcut ARGUS's real dispatch chain
    (`if self.op == "mean": ...`) deliberately does not take."""
    import statistics
    return eval(f"statistics.{op}(vals)")
