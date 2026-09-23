import numpy as np

def compute_errors(n):
    dx = 2.0 * np.pi / n
    # Grid: x_j = j*dx, y_l = l*dx
    # row l is y, col j is x
    y = np.arange(n) * dx
    x = np.arange(n) * dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    g = np.sin(3.0 * X) * np.cos(2.0 * Y)
    
    # Analytic derivatives
    dx_exact = 3.0 * np.cos(3.0 * X) * np.cos(2.0 * Y)
    dxx_exact = -9.0 * g
    dxdy_exact = -6.0 * np.cos(3.0 * X) * np.sin(2.0 * Y)
    lap_exact = -13.0 * g
    
    # Second-order centered finite differences with periodic wrapping
    # dx: axis=1 (columns)
    dx_fd = (np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / (2.0 * dx)
    dxx_fd = (np.roll(g, -1, axis=1) - 2.0 * g + np.roll(g, 1, axis=1)) / (dx * dx)
    dy_fd = (np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) / (2.0 * dx)
    dyy_fd = (np.roll(g, -1, axis=0) - 2.0 * g + np.roll(g, 1, axis=0)) / (dx * dx)
    dxdy_fd = (np.roll(dy_fd, -1, axis=1) - np.roll(dy_fd, 1, axis=1)) / (2.0 * dx)
    lap_fd = dxx_fd + dyy_fd
    
    err_dx_fd = np.max(np.abs(dx_fd - dx_exact))
    err_dxx_fd = np.max(np.abs(dxx_fd - dxx_exact))
    err_dxdy_fd = np.max(np.abs(dxdy_fd - dxdy_exact))
    err_lap_fd = np.max(np.abs(lap_fd - lap_exact))
    
    # Fourier differentiation
    kx = np.zeros(n)
    ky = np.zeros(n)
    half = n // 2
    for j in range(n):
        kx[j] = j if j < half else (-(half) if j == half else j - n)
    for l in range(n):
        ky[l] = l if l < half else (-(half) if l == half else l - n)
        
    g_hat = np.fft.fft2(g)
    
    # Fourier dx
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    dx_four = np.fft.ifft2(1j * KX * g_hat).real
    dxx_four = np.fft.ifft2(- (KX**2) * g_hat).real
    dxdy_four = np.fft.ifft2(- KX * KY * g_hat).real
    lap_four = np.fft.ifft2(- (KX**2 + KY**2) * g_hat).real
    
    err_dx_four = np.max(np.abs(dx_four - dx_exact))
    err_dxx_four = np.max(np.abs(dxx_four - dxx_exact))
    err_dxdy_four = np.max(np.abs(dxdy_four - dxdy_exact))
    err_lap_four = np.max(np.abs(lap_four - lap_exact))
    
    return {
        'fd': (err_dx_fd, err_dxx_fd, err_dxdy_fd, err_lap_fd),
        'four': (err_dx_four, err_dxx_four, err_dxdy_four, err_lap_four)
    }

res32 = compute_errors(32)
res64 = compute_errors(64)

names = [r'dx g', r'dxx g', r'dxdy g', r'Laplacian g']
syms = [r'\partial_x g', r'\partial_x^2 g', r'\partial_x\partial_y g', r'\nabla^2 g']

print(f"{'Derivative':<18} | {'Finite difference, N = 32':<26} | {'Finite difference, N = 64':<26} | {'Ratio (32/64)':<14} | {'Fourier, N = 32':<16}")
print("-" * 110)
for idx in range(4):
    fd32 = res32['fd'][idx]
    fd64 = res64['fd'][idx]
    ratio = fd32 / fd64
    four32 = res32['four'][idx]
    four_str = f"{four32:.2e}" if four32 >= 1e-10 else "< 10^-10"
    print(f"{names[idx]:<18} | {fd32:<26.5f} | {fd64:<26.5f} | {ratio:<14.2f} | {four_str:<16}")
