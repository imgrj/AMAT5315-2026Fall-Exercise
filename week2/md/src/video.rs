//! Minimal raster renderer and FFmpeg encoder for positions beside g(r).

use std::path::Path;

use crate::rdf::Rdf;
use crate::trajectory::Trajectory;

const PANEL: usize = 360;
const WIDTH: usize = 2 * PANEL;
const HEIGHT: usize = PANEL;

fn set(pixel: &mut [u8], x: isize, y: isize, color: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as usize) < WIDTH && (y as usize) < HEIGHT {
        let index = (y as usize * WIDTH + x as usize) * 3;
        pixel[index..index + 3].copy_from_slice(&color);
    }
}

fn line(pixel: &mut [u8], a: (f64, f64), b: (f64, f64), color: [u8; 3]) {
    let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil() as usize).max(1);
    for step in 0..=steps {
        let f = step as f64 / steps as f64;
        set(
            pixel,
            (a.0 + f * (b.0 - a.0)).round() as isize,
            (a.1 + f * (b.1 - a.1)).round() as isize,
            color,
        );
    }
}

fn disk(pixel: &mut [u8], center: (f64, f64), radius: f64, color: [u8; 3]) {
    let limit = radius.ceil() as isize;
    for dy in -limit..=limit {
        for dx in -limit..=limit {
            if (dx * dx + dy * dy) as f64 <= radius * radius {
                set(pixel, center.0.round() as isize + dx, center.1.round() as isize + dy, color);
            }
        }
    }
}

pub fn render_frame(
    box_size: [f64; 2],
    positions: &[[f64; 2]],
    rdf: &[(f64, f64)],
) -> Vec<u8> {
    let mut pixels = vec![248u8; WIDTH * HEIGHT * 3];
    let margin = 22.0;
    let available = PANEL as f64 - 2.0 * margin;
    let scale = (available / box_size[0]).min(available / box_size[1]);
    let x_offset = margin + 0.5 * (available - box_size[0] * scale);
    let y_offset = margin + 0.5 * (available - box_size[1] * scale);
    for p in positions {
        let center = (
            x_offset + p[0].rem_euclid(box_size[0]) * scale,
            y_offset + p[1].rem_euclid(box_size[1]) * scale,
        );
        disk(&mut pixels, center, (0.35 * scale).clamp(2.0, 8.0), [33, 47, 69]);
    }
    let left_a = (x_offset, y_offset);
    let left_b = (x_offset + box_size[0] * scale, y_offset + box_size[1] * scale);
    line(&mut pixels, left_a, (left_b.0, left_a.1), [130, 140, 155]);
    line(&mut pixels, left_a, (left_a.0, left_b.1), [130, 140, 155]);
    line(&mut pixels, left_b, (left_b.0, left_a.1), [130, 140, 155]);
    line(&mut pixels, left_b, (left_a.0, left_b.1), [130, 140, 155]);

    let x0 = PANEL as f64 + margin;
    let y0 = margin;
    let side = available;
    let r_max = 0.5 * box_size[0].min(box_size[1]);
    let g_max = 5.0;
    line(&mut pixels, (x0, y0 + side), (x0 + side, y0 + side), [130, 140, 155]);
    line(&mut pixels, (x0, y0), (x0, y0 + side), [130, 140, 155]);
    line(&mut pixels, (x0, y0 + side - side / g_max), (x0 + side, y0 + side - side / g_max), [205, 210, 218]);
    for pair in rdf.windows(2) {
        let point = |(r, g): (f64, f64)| {
            (
                x0 + r.min(r_max) / r_max * side,
                y0 + side - g.clamp(0.0, g_max) / g_max * side,
            )
        };
        line(&mut pixels, point(pair[0]), point(pair[1]), [13, 110, 253]);
    }

    let mut result = format!("P6\n{WIDTH} {HEIGHT} 255\n").into_bytes();
    result.extend_from_slice(&pixels);
    result
}

pub fn write_video(trajectory: &Trajectory, output: &Path, fps: f64) -> Result<(), String> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let frames_dir = std::env::temp_dir().join(format!("md-video-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&frames_dir).map_err(|e| format!("create {}: {e}", frames_dir.display()))?;
    let mut rdf = Rdf::new(trajectory.meta.box_size, 0.05);
    let render_result = (|| {
        for (index, frame) in trajectory.frames.iter().enumerate() {
            rdf.add_frame(&frame.pos);
            let ppm = render_frame(trajectory.meta.box_size, &frame.pos, &rdf.curve());
            let path = frames_dir.join(format!("{index:06}.ppm"));
            std::fs::write(&path, ppm).map_err(|e| format!("write {}: {e}", path.display()))?;
        }
        let input = frames_dir.join("%06d.ppm");
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-framerate", &fps.to_string(), "-i"])
            .arg(&input)
            .args(["-c:v", "libx264", "-preset", "medium", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(output)
            .status()
            .map_err(|e| format!("start ffmpeg: {e}"))?;
        if status.success() { Ok(()) } else { Err("ffmpeg failed".into()) }
    })();
    let _ = std::fs::remove_dir_all(&frames_dir);
    render_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_has_expected_header_and_size() {
        let ppm = render_frame([10.0, 8.0], &[[1.0, 2.0]], &[(0.1, 0.0), (1.0, 2.0)]);
        let header = format!("P6\n{WIDTH} {HEIGHT} 255\n");
        assert!(ppm.starts_with(header.as_bytes()));
        assert_eq!(ppm.len(), header.len() + WIDTH * HEIGHT * 3);
    }
}
