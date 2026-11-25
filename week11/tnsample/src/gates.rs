//! The seven gate gadgets of the Week 11 sheet, with bits as spins `s = 2x - 1`.
//!
//! Energies are the sheet's own expressions:
//!
//! ```text
//! AND(a,b,z)    s_a s_b - 2 s_a s_z - 2 s_b s_z - s_a - s_b + 2 s_z
//! OR(a,b,z)     s_a s_b - 2 s_a s_z - 2 s_b s_z + s_a + s_b - 2 s_z
//! NOT(a,z)      s_a s_z
//! COPY(a,z)     -s_a s_z
//! SET0(z)       s_z
//! SET1(z)       -s_z
//! XOR(a,b,z,w)  s_a s_b - s_a s_z - 2 s_a s_w - s_b s_z - 2 s_b s_w + 2 s_z s_w
//!               - s_a - s_b + s_z + 2 s_w
//! ```

pub fn spin(bit: usize) -> f64 {
    debug_assert!(bit <= 1);
    2.0 * bit as f64 - 1.0
}

pub fn and(a: f64, b: f64, z: f64) -> f64 {
    a * b - 2.0 * a * z - 2.0 * b * z - a - b + 2.0 * z
}

pub fn or(a: f64, b: f64, z: f64) -> f64 {
    a * b - 2.0 * a * z - 2.0 * b * z + a + b - 2.0 * z
}

pub fn not(a: f64, z: f64) -> f64 {
    a * z
}

pub fn copy(a: f64, z: f64) -> f64 {
    -a * z
}

pub fn set0(z: f64) -> f64 {
    z
}

pub fn set1(z: f64) -> f64 {
    -z
}

pub fn xor(a: f64, b: f64, z: f64, w: f64) -> f64 {
    a * b - a * z - 2.0 * a * w - b * z - 2.0 * b * w + 2.0 * z * w - a - b + z + 2.0 * w
}

/// One named gadget: its bit names, and the bits it may use a helper for.
pub struct Gate {
    pub name: &'static str,
    pub bits: &'static [&'static str],
    pub helper: Option<&'static str>,
}

pub const GATES: [Gate; 7] = [
    Gate {
        name: "AND",
        bits: &["a", "b", "z"],
        helper: None,
    },
    Gate {
        name: "OR",
        bits: &["a", "b", "z"],
        helper: None,
    },
    Gate {
        name: "NOT",
        bits: &["a", "z"],
        helper: None,
    },
    Gate {
        name: "COPY",
        bits: &["a", "z"],
        helper: None,
    },
    Gate {
        name: "SET0",
        bits: &["z"],
        helper: None,
    },
    Gate {
        name: "SET1",
        bits: &["z"],
        helper: None,
    },
    Gate {
        name: "XOR",
        bits: &["a", "b", "z"],
        helper: Some("w"),
    },
];

pub fn energy(name: &str, bits: &[f64], helper: Option<f64>) -> f64 {
    match name {
        "AND" => and(bits[0], bits[1], bits[2]),
        "OR" => or(bits[0], bits[1], bits[2]),
        "NOT" => not(bits[0], bits[1]),
        "COPY" => copy(bits[0], bits[1]),
        "SET0" => set0(bits[0]),
        "SET1" => set1(bits[0]),
        "XOR" => {
            let w = helper.expect("XOR needs the helper spin w");
            xor(bits[0], bits[1], bits[2], w)
        }
        other => panic!("unknown gate {other}"),
    }
}

/// The circuit `AND(a,b,t) + NOT(t,out) + SET1(out)` of the sheet's Figure 10.
pub fn circuit(a: f64, b: f64, t: f64, o: f64) -> f64 {
    and(a, b, t) + not(t, o) + set1(o)
}

/// The gate's rule as a predicate on its bits, used to mark allowed rows.
pub fn allowed(name: &str, bits: &[usize]) -> bool {
    match name {
        "AND" => bits[2] == (bits[0] & bits[1]),
        "OR" => bits[2] == (bits[0] | bits[1]),
        "NOT" => bits[1] == (1 - bits[0]),
        "COPY" => bits[1] == bits[0],
        "SET0" => bits[0] == 0,
        "SET1" => bits[0] == 1,
        "XOR" => bits[2] == (bits[0] ^ bits[1]),
        other => panic!("unknown gate {other}"),
    }
}

/// The circuit's rule: `t = a AND b`, `out = NOT t`, and `SET1` requires `out = 1`.
pub fn circuit_allowed(bits: &[usize]) -> bool {
    let (a, b, t, o) = (bits[0], bits[1], bits[2], bits[3]);
    t == (a & b) && o == (1 - t) && o == 1
}
