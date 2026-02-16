import random
from deap import gp
import inspect

def gen_safe(pset, min_, max_, type_):
    """
    Generates a tree ensuring that we can always terminate even if the root type
    has no direct terminals (by falling back to non-recursive primitives).
    
    This handles the case where types like Ephemeral have no terminal but
    can be produced by primitives like rf_classification.
    """
    def generate(type_, depth):
        # Check if we have terminals and primitives for this type
        terminals = pset.terminals.get(type_, [])
        primitives = pset.primitives.get(type_, [])
        
        def pick_terminal():
            """Helper to properly instantiate a terminal (handles ephemeral constants)."""
            term = random.choice(terminals)
            # Ephemeral constants in DEAP are classes that inherit from a special base
            # They have a 'func' attribute and need to be instantiated
            if inspect.isclass(term):
                # This is an ephemeral constant class - instantiate it
                return [term()]
            else:
                # Regular terminal instance
                return [term]
        
        def pick_primitive_and_recurse(available_depth):
            """Helper to pick a primitive and recursively generate its arguments."""
            prim = random.choice(primitives)
            expr = [prim]
            for arg_type in prim.args:
                expr.extend(generate(arg_type, available_depth - 1))
            return expr

        # If we MUST stop (depth limit reached or below)
        if depth <= 0:
            if terminals:
                return pick_terminal()
            else:
                # No terminals - must pick a non-recursive primitive
                # Filter for primitives that don't take 'type_' as argument (to avoid infinite recursion)
                safe_prims = [p for p in primitives if type_ not in p.args]
                
                if not safe_prims:
                    raise ValueError(f"Cannot terminate type {type_}: No terminals and no safe primitives.")
                
                prim = random.choice(safe_prims)
                expr = [prim]
                for arg_type in prim.args:
                    # Force termination for children
                    expr.extend(generate(arg_type, 0))
                return expr

        # If we can continue growing
        else:
            if not terminals:
                # No terminals for this type - must pick primitive
                if not primitives:
                    raise ValueError(f"No primitives or terminals for type {type_}")
                return pick_primitive_and_recurse(depth)
            else:
                # We have terminals - choose between terminal and primitive
                # Use genGrow-like behavior: uniform probability between all options
                n_terms = len(terminals)
                n_prims = len(primitives)
                
                if n_prims == 0 or random.random() < (n_terms / (n_terms + n_prims)):
                    return pick_terminal()
                else:
                    return pick_primitive_and_recurse(depth)

    return generate(type_, max_)
