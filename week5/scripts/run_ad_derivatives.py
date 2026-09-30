import json
from pathlib import Path
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

def main():
    r = 1.3
    
    # Forward mode by hand, node by node
    # Node 0: r
    dot_r = 1.0
    
    # Node 1: a = r^-6
    # jax.jvp of node a
    a, dot_a = jax.jvp(lambda x: x**(-6), (r,), (dot_r,))
    
    # Node 2: b = a^2
    b, dot_b = jax.jvp(lambda x: x**2, (a,), (dot_a,))
    
    # Node 3: c = b - a
    c, dot_c = jax.jvp(lambda x, y: x - y, (b, a), (dot_b, dot_a))
    
    # Node 4: U = 4c
    U, dot_U = jax.jvp(lambda x: 4.0 * x, (c,), (dot_c,))
    
    # Reverse mode by hand, node by node
    # Primal values needed: r and a
    bar_U = 1.0
    
    # Step from U to c: U = 4c
    _, vjp_U = jax.vjp(lambda x: 4.0 * x, c)
    (bar_c,) = vjp_U(bar_U)
    
    # Step from c to b and a: c = b - a
    _, vjp_c = jax.vjp(lambda x, y: x - y, b, a)
    bar_b, bar_a_from_c = vjp_c(bar_c)
    
    # Step from b to a: b = a^2
    _, vjp_b = jax.vjp(lambda x: x**2, a)
    (bar_a_from_b,) = vjp_b(bar_b)
    
    # Combine adjoint contributions to a
    bar_a = bar_a_from_c + bar_a_from_b
    
    # Step from a to r: a = r^-6
    _, vjp_a = jax.vjp(lambda x: x**(-6), r)
    (bar_r,) = vjp_a(bar_a)
    
    # JAX own grad function
    def pair_energy(x):
        a_val = x**(-6)
        b_val = a_val**2
        c_val = b_val - a_val
        return 4.0 * c_val
        
    grad_val = float(jax.grad(pair_energy)(r))
    
    out_dir = Path("artifacts/ad")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        "r": float(r),
        "energy": float(U),
        "tangents": {
            "r": float(dot_r),
            "a": float(dot_a),
            "b": float(dot_b),
            "c": float(dot_c),
            "U": float(dot_U)
        },
        "adjoints": {
            "U": float(bar_U),
            "c": float(bar_c),
            "b": float(bar_b),
            "a": float(bar_a),
            "r": float(bar_r)
        },
        "jax_grad": grad_val
    }
    
    out_path = out_dir / "derivatives.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved {out_path}")
    print(f"r: {r}")
    print(f"energy: {float(U):.16f}")
    print(f"tangents.U: {float(dot_U):.16f}")
    print(f"adjoints.r: {float(bar_r):.16f}")
    print(f"adjoints.a: {float(bar_a):.16f}")
    print(f"jax_grad: {grad_val:.16f}")

if __name__ == "__main__":
    main()
