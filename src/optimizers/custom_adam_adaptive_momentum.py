import torch

def custom_adam_adaptive_momentum(
    params,
    grads,
    state,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    k=0.5
):
    """
    Custom Adam variant with adaptive momentum scaling.
    Momentum is boosted when gradient directions remain consistent,
    and reduced when gradients fluctuate (high 'noise metric').
    """

    beta1, beta2 = betas

    for p, g in zip(params, grads):
        if g is None:
            continue
        p_data = p.data
        g_data = g.data
        if weight_decay != 0:
            g_data = g_data.add(p_data, alpha=weight_decay)

        st = state.setdefault(p, {})
        m = st.get('m', torch.zeros_like(p_data))
        v = st.get('v', torch.zeros_like(p_data))
        t = st.get('t', 0) + 1
        prev_g = st.get('prev_g', torch.zeros_like(p_data))

        # Adam-style exponential moving averages
        m_new = beta1 * m + (1 - beta1) * g_data
        v.mul_(beta2).addcmul_(g_data, g_data, value=(1 - beta2))

        # ---- Noise metric (gradient consistency) ----
        diff = g_data - prev_g
        sign_consistency = (g_data * prev_g).clamp(min=0)
        noise_metric = (diff.abs() / (g_data.abs() + 1e-8)) * (
            1 - sign_consistency / (g_data.abs() + 1e-8)
        )

        # Adaptive momentum scaling
        adaptive_scale = 1 + k * torch.clamp(1 - noise_metric.mean(), 0.0, 2.0)
        m_scaled = m_new * adaptive_scale

        # Bias corrections
        m_hat = m_scaled / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update parameters
        p_data.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)

        # Save updated state
        st['m'] = m_new
        st['v'] = v
        st['t'] = t
        st['prev_g'] = g_data.clone()
        state[p] = st
