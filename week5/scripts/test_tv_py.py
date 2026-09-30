import math

def binomial(n, k):
    return math.comb(n, k)

def binomial_fit(n_steps, delta):
    tau = 1
    while n_steps > binomial(tau + delta, delta):
        tau += 1
    return tau

def mid(delta, tau, sigma, phi):
    den = tau + delta
    if den == 0:
        return sigma
    num = delta * sigma + tau * phi
    kappa = (num + den - 1) // den
    if kappa >= phi and delta > 0:
        kappa = max(sigma + 1, phi - 1)
    return kappa

def build_treeverse_schedule(n_steps, delta):
    actions = []
    saved_count = 1
    peak_saved = 1
    forward_calls = 0
    reverse_calls = 0
    working_pos = 0

    def record(act, step):
        nonlocal saved_count, peak_saved, forward_calls, reverse_calls
        if act == 'store':
            saved_count += 1
        elif act == 'fetch':
            saved_count -= 1
        elif act == 'call':
            forward_calls += 1
        elif act == 'grad':
            reverse_calls += 1
        if saved_count > peak_saved:
            peak_saved = saved_count
        actions.append({'action': act, 'step': step, 'saved_states': saved_count})

    def recurse(delta_cur, tau_cur, beta, sigma, phi):
        nonlocal working_pos
        if sigma > beta:
            effective_delta = max(0, delta_cur - 1)
            if working_pos != beta:
                record('restore', beta)
                working_pos = beta
            for j in range(beta, sigma):
                record('call', j)
                working_pos = j + 1
            record('store', sigma)
        else:
            effective_delta = delta_cur

        kappa = mid(effective_delta, tau_cur, sigma, phi)
        while tau_cur > 0 and kappa < phi:
            recurse(effective_delta, tau_cur, sigma, kappa, phi)
            tau_cur -= 1
            phi = kappa
            if tau_cur > 0:
                kappa = mid(effective_delta, tau_cur, sigma, phi)

        record('grad', sigma)
        if sigma > beta:
            record('fetch', sigma)

    tau = binomial_fit(n_steps, delta)
    recurse(delta, tau, 0, 0, n_steps)
    return {
        'actions': actions,
        'forward_calls': forward_calls,
        'reverse_calls': reverse_calls,
        'peak_saved_states': peak_saved
    }

if __name__ == '__main__':
    for d in [1, 3, 5, 10]:
        s = build_treeverse_schedule(240, d)
        print(f"delta={d:2d}: peak={s['peak_saved_states']}, forward_calls={s['forward_calls']}, reverse_calls={s['reverse_calls']}, actions={len(s['actions'])}")

    s_m = build_treeverse_schedule(1200, 5)
    print(f"Marmousi delta=5: peak={s_m['peak_saved_states']}, forward_calls={s_m['forward_calls']}, reverse_calls={s_m['reverse_calls']}, actions={len(s_m['actions'])}")
