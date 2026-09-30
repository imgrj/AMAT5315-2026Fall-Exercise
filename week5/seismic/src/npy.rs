use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

pub fn write_npy_f64<P: AsRef<Path>>(path: P, shape: &[usize], data: &[f64]) -> io::Result<()> {
    let mut file = File::create(path)?;
    let mut header = String::from("{'descr': '<f8', 'fortran_order': False, 'shape': (");
    for (i, &s) in shape.iter().enumerate() {
        if i > 0 {
            header.push_str(", ");
        }
        header.push_str(&s.to_string());
    }
    if shape.len() == 1 {
        header.push(',');
    }
    header.push_str("), }");

    // Total preamble before header is 10 bytes:
    // 6 bytes magic (\x93NUMPY), 1 byte major (1), 1 byte minor (0), 2 bytes header_len (u16 LE)
    // We pad header so (10 + header_len) is a multiple of 64 bytes.
    let unpadded_len = header.len() + 1; // +1 for '\n'
    let pad_len = (64 - ((10 + unpadded_len) % 64)) % 64;
    header.extend(std::iter::repeat(' ').take(pad_len));
    header.push('\n');

    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u16;

    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;

    // Write data bytes
    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f64>())
    };
    file.write_all(byte_slice)?;
    file.flush()?;
    Ok(())
}

pub fn write_npy_f32<P: AsRef<Path>>(path: P, shape: &[usize], data: &[f32]) -> io::Result<()> {
    let mut file = File::create(path)?;
    let mut header = String::from("{'descr': '<f4', 'fortran_order': False, 'shape': (");
    for (i, &s) in shape.iter().enumerate() {
        if i > 0 {
            header.push_str(", ");
        }
        header.push_str(&s.to_string());
    }
    if shape.len() == 1 {
        header.push(',');
    }
    header.push_str("), }");

    let unpadded_len = header.len() + 1;
    let pad_len = (64 - ((10 + unpadded_len) % 64)) % 64;
    header.extend(std::iter::repeat(' ').take(pad_len));
    header.push('\n');

    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u16;

    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;

    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f32>())
    };
    file.write_all(byte_slice)?;
    file.flush()?;
    Ok(())
}

pub fn read_npy_f64<P: AsRef<Path>>(path: P) -> io::Result<(Vec<usize>, Vec<f64>)> {
    let mut file = File::open(path)?;
    let mut magic_version = [0u8; 8];
    file.read_exact(&mut magic_version)?;
    if &magic_version[0..6] != b"\x93NUMPY" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid NPY magic"));
    }

    let mut hlen_bytes = [0u8; 2];
    file.read_exact(&mut hlen_bytes)?;
    let header_len = u16::from_le_bytes(hlen_bytes) as usize;

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header_str = String::from_utf8_lossy(&header_buf);

    // Extract shape: between 'shape': ( and )
    let shape_start = header_str.find("'shape': (")
        .or_else(|| header_str.find("\"shape\": ("))
        .or_else(|| header_str.find("'shape':("))
        .or_else(|| header_str.find("\"shape\":("))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Could not find shape in NPY header"))?;

    let open_paren = header_str[shape_start..].find('(').unwrap() + shape_start;
    let close_paren = header_str[open_paren..].find(')').ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Could not find closing paren in NPY shape")
    })? + open_paren;

    let shape_sub = &header_str[open_paren + 1..close_paren];
    let shape: Vec<usize> = shape_sub
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)))
        .collect::<Result<Vec<_>, _>>()?;

    let total_elements: usize = shape.iter().product();
    let mut data = vec![0.0f64; total_elements];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total_elements * std::mem::size_of::<f64>())
    };
    file.read_exact(byte_slice)?;

    Ok((shape, data))
}
