import numpy as np
import torch
import math
from tqdm import tqdm
import zstandard
import pickle
from datetime import datetime, timezone
import signal
import sys


# Set device: GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)
dtype = torch.float64

# Global flag to record that SIGINT was received.
interrupted = False


def sigint_handler(*_):
    global interrupted
    interrupted = True
    print("SIGINT received; finishing current work and saving before exit.")


def write_pkl(data, file, **kwargs):
    with open(file, 'wb') as f:
        cctx = zstandard.ZstdCompressor(write_checksum=True, threads=-1, **kwargs)
        with cctx.stream_writer(f) as compressor:
            pickle.dump(data, compressor)


def drazin_inverse_MP(W, p_ss):
    I = torch.eye(len(p_ss), device=W.device, dtype=W.dtype)
    P = I - torch.outer(p_ss, torch.ones_like(p_ss))
    W_D = P @ torch.linalg.pinv(W) @ P
    return W_D


def build_rate_matrices(N, f_x, f_y, a_x, a_y, Gamma_x, Gamma_y, chi_x=0.0, chi_y=0.0):
    """
    Build the counting-field dependent rate matrix (W^phi) and its derivatives.
    The state is indexed by i = 0,...,N-1 (with periodic BCs).
    The off-diagonals are only for i->i+1 and i->i-1.

    Returns:
      M: The N x N matrix at given chi_x, chi_y.
      Jx1, Jy1: First derivatives (N x N) with respect to chi_x and chi_y.
      Jx2, Jy2: Second derivatives (N x N) with respect to chi_x and chi_y.
      M0: The matrix at chi=0 (used for the stationary state).
      rates_x_up, rates_x_down: The oscillator X rates (without counting field factors).
    """
    M = torch.zeros((N, N), dtype=dtype, device=device)
    Jx1 = torch.zeros_like(M)
    Jy1 = torch.zeros_like(M)
    Jx2 = torch.zeros_like(M)
    Jy2 = torch.zeros_like(M)

    # Precompute factors for counting fields (exponentials)
    exp_px = math.exp(2 * math.pi * chi_x)
    exp_mx = math.exp(-2 * math.pi * chi_x)
    exp_py = math.exp(-2 * math.pi * chi_y)
    exp_my = math.exp(2 * math.pi * chi_y)

    # We also need the rates at chi=0 for the diagonal elements.
    rates_x_up = torch.zeros(N, dtype=dtype, device=device)
    rates_x_down = torch.zeros(N, dtype=dtype, device=device)
    rates_y_up = torch.zeros(N, dtype=dtype, device=device)
    rates_y_down = torch.zeros(N, dtype=dtype, device=device)

    for i in range(N):
        # For oscillator X:
        # Transition from state i -> i+1:
        # argument in sine: 2π*(i) / N   (since i - 1/2 + 1/2 = i)
        rate_x_up = N * Gamma_x * (1 - a_x * math.sin(2 * math.pi * i / N)) / (1 + math.exp(-f_x))
        # Transition from state i -> i-1:
        # argument: 2π*(i-1)/N  (since i - 1/2 - 1/2 = i-1)
        rate_x_down = N * Gamma_x * (1 - a_x * math.sin(2 * math.pi * ((i - 1) % N) / N)) / (
                    1 + math.exp(f_x))

        rates_x_up[i] = rate_x_up
        rates_x_down[i] = rate_x_down

        # For oscillator Y:
        # For upward jump (i -> i+1):
        # In the effective formulation, eq. (7b) yields:
        # argument: 2π*(-i - 1) / N and denominator 1+exp(f_y).
        # Using sin(-θ) = - sin(θ):
        rate_y_up = N * Gamma_y * (1 + a_y * math.sin(2 * math.pi * ((i + 1) % N) / N)) / (
                    1 + math.exp(f_y))
        # For downward jump (i -> i-1):
        # argument: 2π*(-i) / N => 1 - a_y * sin(2π*(-i)/N) = 1 + a_y*sin(2π*i/N)
        rate_y_down = N * Gamma_y * (1 + a_y * math.sin(2 * math.pi * i / N)) / (1 + math.exp(-f_y))

        rates_y_up[i] = rate_y_up
        rates_y_down[i] = rate_y_down

    # Build off-diagonal elements.
    # For each state i, the only nonzero off-diagonals are from i to i+1 and i to i-1.
    for i in range(N):
        ip = (i + 1) % N  # periodic BC
        im = (i - 1) % N

        # For upward (i -> i+1):
        # Counting-field dressed contributions:
        # X: multiplies e^(+2π chi_x); Y: multiplies e^(-2π chi_y)
        rate_up = rates_x_up[i] * exp_px + rates_y_up[i] * exp_py
        M[ip, i] = rate_up
        # Derivatives w.r.t chi_x and chi_y:
        Jx1[ip, i] = 2 * math.pi * rates_x_up[i] * exp_px
        Jx2[ip, i] = (2 * math.pi) ** 2 * rates_x_up[i] * exp_px
        Jy1[ip, i] = -2 * math.pi * rates_y_up[i] * exp_py
        Jy2[ip, i] = (2 * math.pi) ** 2 * rates_y_up[i] * exp_py

        # For downward (i -> i-1):
        # X: multiplies e^(-2π chi_x); Y: multiplies e^(+2π chi_y)
        rate_down = rates_x_down[i] * exp_mx + rates_y_down[i] * exp_my
        M[im, i] = rate_down
        Jx1[im, i] = -2 * math.pi * rates_x_down[i] * exp_mx
        Jx2[im, i] = (2 * math.pi) ** 2 * rates_x_down[i] * exp_mx
        Jy1[im, i] = 2 * math.pi * rates_y_down[i] * exp_my
        Jy2[im, i] = (2 * math.pi) ** 2 * rates_y_down[i] * exp_my

    # Diagonal elements: they are fixed to the chi=0 rates.
    # (i.e. use the rates computed with exp=1)
    M0 = torch.zeros((N, N), dtype=dtype, device=device)
    for i in range(N):
        rate_up_0 = rates_x_up[i] + rates_y_up[i]
        rate_down_0 = rates_x_down[i] + rates_y_down[i]
        M0[i, i] = - (rate_up_0 + rate_down_0)
        # Set the same diagonal in the dressed matrix (they are independent of chi)
        M[i, i] = M0[i, i]

    return M, M0, Jx1, Jy1, Jx2, Jy2, rates_x_up, rates_x_down


def compute_drazin(M, Jx1, Jy1, Jx2, Jy2, p_ss, N):
    ones = torch.ones((1, N), dtype=M.dtype, device=M.device)
    W_D = drazin_inverse_MP(M, p_ss.squeeze())
    cov_x = ones @ (Jx2 @ p_ss - 2 * Jx1 @ W_D @ Jx1 @ p_ss) / N
    cov_y = ones @ (Jy2 @ p_ss - 2 * Jy1 @ W_D @ Jy1 @ p_ss) / N
    cov_xy = - ones @ (Jx1 @ W_D @ Jy1 @ p_ss + Jy1 @ W_D @ Jx1 @ p_ss) / N
    return cov_x, cov_y, cov_xy


def solve_oscillator_1d(N, f_x, f_y, a_x, a_y, Gamma_x, Gamma_y):
    """
    Solve for the stationary state and compute the following quantities:
      - Stationary distribution p_ss (null vector of M0).
      - Average phase velocities omega_x and omega_y.
      - (Phase) covariance for oscillator X (using the spectral approach and Drazin inverse).
      - Mutual information I_{XY}.
      - Information flow from Y to X.

    Returns a dictionary with these quantities.
    """
    # Build the rate matrix at chi=0 and also get the derivative matrices.
    M, M0, Jx1, Jy1, Jx2, Jy2, rates_x_up, rates_x_down = build_rate_matrices(
        N, f_x, f_y, a_x, a_y, Gamma_x, Gamma_y, chi_x=0.0, chi_y=0.0
    )

    # Find the null vector of M0 using SVD.
    # (We expect a single zero singular value.)
    _, _, Vh = torch.linalg.svd(M)
    p_ss = Vh[-1].clone().detach().abs()  # take the last row of Vh
    p_ss = p_ss / p_ss.sum()  # normalize to 1
    p_ss = p_ss.reshape(N, 1)  # column vector
    ones = torch.ones((1, N), dtype=dtype, device=device)

    # Average phase velocities (using the derivative matrices)
    omega_x = (ones @ (Jx1 @ p_ss)) / N
    omega_y = (ones @ (Jy1 @ p_ss)) / N
    cov_x, cov_y, cov_xy = compute_drazin(M, Jx1, Jy1, Jx2, Jy2, p_ss, N)
    

    # Mutual information (using the effective probability distribution):
    p_ss_vec = p_ss.flatten()
    i_xy = math.log(N) + torch.sum(p_ss_vec * torch.log(p_ss_vec + 1e-15))

    # Information flow from Y to X.
    # According to the effective formulation,
    # i_flow = sum_i [ W^(x)_{i+1,i} p_i - W^(x)_{i,i+1} p_{i+1} ] * ln (p_{i+1}/p_i)
    p_roll = torch.roll(p_ss_vec, -1)
    term = (rates_x_up * p_ss_vec - torch.roll(rates_x_down, -1) * p_roll)
    # To avoid log(0) issues, add a small constant.
    i_flow = torch.sum(term * torch.log(1e-15 + p_roll / (p_ss_vec + 1e-15)))

    results = {
        'p_ss': p_ss_vec.cpu().numpy(),
        'omega_x': omega_x.item(),
        'omega_y': omega_y.item(),
        'cov_x': cov_x.item(),
        'cov_y': cov_y.item(),
        'cov_xy': cov_xy.item(),
        'i_xy': i_xy.item(),
        'i_flow': i_flow.item(),
        'params': {
            'N': int(N),
            'f_x': float(f_x), 'f_y': float(f_y),
            'a_x': float(a_x), 'a_y': float(a_y),
            'Gamma_x': float(Gamma_x), 'Gamma_y': float(Gamma_y),
        },
    }
    return results


class BreakLoop(Exception):
    pass


if __name__ == '__main__':
    # Define fixed parameters.
    f_y = 2.0
    a_x = 0.5
    a_y = 0.5
    Gamma_x = 1.0
    Gamma_y = 1.0
    N_values = [40,160]
    f_x_values = np.arange(-3.0, 3.001, 0.05)
    total_cases = len(N_values) * len(f_x_values)
    pbar = tqdm(total=total_cases, smoothing=1)

    # Set our custom SIGINT handler.
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        all_results = []
        # For each N and each f_x, compute the quantities.
        for N in N_values:
            all_results.append([])
            for f_x in f_x_values:
                pbar.set_description(f"N={N}, fx={f_x:+.4f}")
                results = solve_oscillator_1d(N, f_x, f_y, a_x, a_y, Gamma_x, Gamma_y)
                all_results[-1].append(results)
                pbar.update()
                if interrupted:
                    raise BreakLoop
    except BreakLoop:
        pass
    except Exception:
        # Let other exceptions propagate but ensure saving happens.
        raise
    finally:
        pbar.close()
        # Temporarily ignore SIGINT during the saving routine.
        temp_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        datetime_str = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        write_pkl(all_results, f"one_d_results/{datetime_str}.pkl.zstd")
        # Restore the original SIGINT handler.
        signal.signal(signal.SIGINT, original_sigint_handler)
        # If an interrupt was flagged, exit after saving.
        if interrupted:
            sys.exit(1)

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    axs = fig.subplots(2, 2, sharex=True, sharey=False)

    ax = axs[0,0]
    for i, (N, results) in enumerate(zip(N_values, all_results)):
        ax.plot(f_x_values, [r['omega_x'] for r in results], f"C{i}", label=f"N={N}")
        ax.plot(f_x_values, [r['omega_y'] for r in results], f"C{i}--")
        ax.plot(f_x_values, [(r['omega_x'] - r['omega_y'])/2 for r in results], f"C{i}-.")
        ax.plot(f_x_values, [(r['omega_x'] + r['omega_y'])/2 for r in results], f"C{i}:")
        ax.set_xlim(f_x_values[0], f_x_values[-1])
        ax.set_title("Phase velocity")
        ax.grid()
        ax.legend()

    ax = axs[0,1]
    for i, (N, results) in enumerate(zip(N_values, all_results)):
        ax.plot(f_x_values, [r['cov_x'] for r in results], f"C{i}", label=f"N={N}")
        ax.plot(f_x_values, [r['cov_y'] for r in results], f"C{i}--")
        ax.plot(f_x_values, [r['cov_xy'] for r in results], f"C{i}-.")
        ax.plot(f_x_values, [r['cov_x'] + r['cov_y'] - 2*r['cov_xy'] for r in results], f"C{i}:")
        ax.set_title("Co/variance")
        ax.set_ylim(-300, 600)
        ax.grid()
        ax.legend()

    ax = axs[1,0]
    for i, (N, results) in enumerate(zip(N_values, all_results)):
        ax.plot(f_x_values, [r['i_xy'] for r in results], f"C{i}", label=f"N={N}")
        ax.plot(f_x_values, [r['i_flow'] for r in results], f"C{i}--")
        ax.set_xlabel("$f_x$")
        ax.set_title("MI")
        ax.grid()
        ax.legend()

    ax = axs[1,1]
    for i, (N, results) in enumerate(zip(N_values, all_results)):
        ax.plot(f_x_values, [r['i_xy'] / np.log(N) for r in results], f"C{i}", label=f"N={N}")
        ax.plot(f_x_values, [r['i_flow'] / np.log(N) for r in results], f"C{i}--")
        ax.set_xlabel("$f_x$")
        ax.set_title(r"MI / log$\,N$")
        ax.grid()
        ax.legend()

    fig.tight_layout()
    plt.show()
