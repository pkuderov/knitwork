# debug_nan.py
"""
Вставить ВРЕМЕННО в run_sdq3.py сразу после создания модели и оптимизатора.
Запускает один синтетический forward+backward и показывает где именно NaN/Inf.
"""

import torch
import torch.nn as nn
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ХУКИ: ловим NaN/Inf в активациях (forward)
# ═══════════════════════════════════════════════════════════════════════════════

class NanHookManager:
    def __init__(self):
        self.hooks = []
        self.first_nan: dict | None = None   # первый пойманный NaN

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            h = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(h)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            if self.first_nan is not None:
                return   # уже нашли — не спамим
            # Проверяем все тензоры выхода
            outs = [out] if isinstance(out, torch.Tensor) else (
                list(out) if isinstance(out, (tuple, list)) else []
            )
            for i, t in enumerate(outs):
                if not isinstance(t, torch.Tensor):
                    continue
                if not torch.isfinite(t).all():
                    self.first_nan = {
                        'module': name,
                        'type':   type(module).__name__,
                        'out_idx': i,
                        'shape':  tuple(t.shape),
                        'n_nan':  int((~torch.isfinite(t)).sum().item()),
                        'min':    float(t[torch.isfinite(t)].min()) if torch.isfinite(t).any() else float('nan'),
                        'max':    float(t[torch.isfinite(t)].max()) if torch.isfinite(t).any() else float('nan'),
                    }
                    # Проверяем входы этого модуля
                    bad_inputs = []
                    inps = [inp] if isinstance(inp, torch.Tensor) else (
                        list(inp) if isinstance(inp, (tuple, list)) else []
                    )
                    for j, ti in enumerate(inps):
                        if isinstance(ti, torch.Tensor) and not torch.isfinite(ti).all():
                            bad_inputs.append(j)
                    self.first_nan['bad_inputs'] = bad_inputs
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ПРОВЕРКА ПАРАМЕТРОВ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

def check_params(model: nn.Module) -> list[str]:
    """Ищем NaN/Inf в весах модели."""
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            n_bad = int((~torch.isfinite(p)).sum().item())
            bad.append(f"  PARAM NaN/Inf: {name} | shape={tuple(p.shape)} | n_bad={n_bad}")
    return bad


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ПРОВЕРКА ГРАДИЕНТОВ — подробно по каждому параметру
# ═══════════════════════════════════════════════════════════════════════════════

def check_grads(model: nn.Module) -> list[str]:
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            n_bad = int((~torch.isfinite(g)).sum().item())
            gmax  = float(g[torch.isfinite(g)].abs().max()) if torch.isfinite(g).any() else float('nan')
            bad.append(
                f"  GRAD NaN/Inf: {name:<60} "
                f"shape={str(tuple(p.shape)):<20} "
                f"n_bad={n_bad:<6} "
                f"max_finite={gmax:.4f}"
            )
    return bad


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ПОШАГОВАЯ ДИАГНОСТИКА ЧЕРЕЗ ВЕСЬ GRID STEP
# ═══════════════════════════════════════════════════════════════════════════════

def debug_grid_step(rnn, x: torch.Tensor, h: torch.Tensor):
    """
    Повторяет логику _grid_step вручную с проверкой на каждом шаге.
    """
    print("\n── debug_grid_step ──────────────────────────────────────────")
    n_t = rnn.n_trainable_cols
    n_r = rnn.n_reservoir_cols

    def check(tag: str, t: torch.Tensor):
        ok = torch.isfinite(t).all()
        mn = float(t.min()) if ok else float('nan')
        mx = float(t.max()) if ok else float('nan')
        print(f"  {'✓' if ok else '✗ NaN/Inf':<12} {tag:<55} "
              f"shape={str(tuple(t.shape)):<25} min={mn:+.4f} max={mx:+.4f}")
        return ok

    check("input x", x)
    check("input h", h)

    # Embedding / input проекции
    x_cols = torch.stack([proj(x) for proj in rnn.col_input_projs], dim=1)
    check("x_cols (after col_input_projs)", x_cols)

    for li in range(rnn.n_layers):
        print(f"\n  ── Layer {li} ──")
        hl = h[li]
        check(f"  h[{li}] (input to layer)", hl)

        if li == 0:
            x_t_batch = x_cols
        else:
            x_t_batch = hl.permute(1, 0, 2)
        check(f"  x_t_batch L{li}", x_t_batch)

        h_t_in = hl[:n_t].permute(1, 0, 2)
        x_t_in = x_t_batch[:, :n_t, :]
        check(f"  h_t_in  L{li}", h_t_in)
        check(f"  x_t_in  L{li}", x_t_in)

        # Trainable: пошагово
        cell = rnn.trainable_cells[li]
        x_t = x_t_in.permute(1, 2, 0)
        h_t = h_t_in.permute(1, 2, 0)

        def gx(W, b, tag):
            out = torch.bmm(W, x_t).permute(0, 2, 1)
            if b is not None:
                out = out + b.unsqueeze(1)
            check(f"  gx({tag}) L{li}", out)
            return out

        def gh(U, h_src, tag):
            out = torch.bmm(U, h_src).permute(0, 2, 1)
            check(f"  gh({tag}) L{li}", out)
            return out

        o_raw = gx(cell.W_o, cell.b_o, "W_o") + gh(cell.U_o, h_t, "U_o")
        check(f"  o_raw L{li}", o_raw)
        o_t = torch.sigmoid(o_raw)
        check(f"  o_t   L{li}", o_t)

        h_p = h_t_in.permute(1, 0, 2)
        oh = (o_t * h_p).permute(0, 2, 1)
        check(f"  o_t*h L{li}", oh)

        c_raw = gx(cell.W_c, cell.b_c, "W_c") + gh(cell.U_c, oh, "U_c")
        check(f"  c_raw L{li}", c_raw)

        if cell.ln_c is not None:
            c_normed = torch.stack([cell.ln_c[i](c_raw[i]) for i in range(cell.n_cols)], dim=0)
            check(f"  c_normed (after LN) L{li}", c_normed)
        else:
            c_normed = c_raw

        c_t = torch.tanh(c_normed)
        check(f"  c_t   L{li}", c_t)

        f_raw = gx(cell.W_f, cell.b_f, "W_f") + gh(cell.U_f, h_t, "U_f")
        check(f"  f_raw L{li}", f_raw)
        betas = cell.betas.view(cell.n_cols, 1, 1)
        lam_t = torch.sigmoid(f_raw) * (1.0 - betas) + betas
        check(f"  lam_t L{li}", lam_t)

        h_t_new_raw = lam_t * h_p + (1.0 - lam_t) * c_t
        check(f"  h_t_new L{li}", h_t_new_raw)
        h_t_new = h_t_new_raw.permute(1, 0, 2)

        if n_r > 0:
            h_r_in = hl[n_t:].permute(1, 0, 2)
            x_r_in = x_t_batch[:, n_t:, :]
            check(f"  h_r_in L{li}", h_r_in)
            check(f"  x_r_in L{li}", x_r_in)
            h_r_new = rnn._batched_reservoir_forward(li, x_r_in, h_r_in)
            check(f"  h_r_new L{li}", h_r_new)

            if rnn.cross_attns is not None:
                h_t_new = rnn.cross_attns[li](h_t_new, h_r_new)
                check(f"  h_t_new after CrossAttn L{li}", h_t_new)

            h_all = torch.cat([h_t_new, h_r_new], dim=1)
        else:
            h_all = h_t_new

        h_all_seq = h_all.permute(1, 0, 2)
        check(f"  h_all_seq L{li}", h_all_seq)

        msg, _ = rnn.attn[li](h_all_seq, return_weights=False)
        check(f"  msg (after MHA) L{li}", msg)

        msg_t   = msg[:n_t]
        h_t_seq = h_t_new.permute(1, 0, 2)
        gate_in = torch.cat([h_t_seq, msg_t], dim=-1)
        check(f"  gate_in L{li}", gate_in)
        gate_logit = rnn.attn_gates[li](gate_in)
        check(f"  gate_logit L{li}", gate_logit)
        g = torch.sigmoid(gate_logit)
        h_t_merged = (1.0 - g) * h_t_seq + g * msg_t
        check(f"  h_t_merged L{li}", h_t_merged)

    print("── end debug_grid_step ──────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ГЛАВНАЯ ФУНКЦИЯ ДИАГНОСТИКИ — вставить в main() сразу после создания rnn
# ═══════════════════════════════════════════════════════════════════════════════

def run_nan_diagnostics(rnn, gen, device, loss_fn):
    print("\n" + "═" * 70)
    print("NaN DIAGNOSTICS START")
    print("═" * 70)

    # 5.1 Проверяем веса модели ДО обучения
    print("\n[1] Параметры модели:")
    bad_params = check_params(rnn)
    if bad_params:
        for s in bad_params:
            print(s)
    else:
        print("  ✓ Все параметры конечны")

    # 5.2 Синтетический входной батч
    bsz = 4
    # Случайные токены в диапазоне vocab
    fake_tokens = torch.randint(0, gen.n_tokens, (bsz, 1), device=device)
    fake_targets = torch.randint(0, gen.V, (bsz,), device=device)
    h = rnn.init_state(bsz)

    print(f"\n[2] Тестовый батч: tokens={tuple(fake_tokens.shape)} targets={tuple(fake_targets.shape)}")
    print(f"    dtype={h.dtype} device={device}")

    # 5.3 Forward с хуками
    print("\n[3] Forward pass с хуками на NaN:")
    hook_mgr = NanHookManager()
    hook_mgr.register(rnn)
    try:
        with torch.autograd.set_detect_anomaly(True):
            y, h_new, extras = rnn(fake_tokens, h, return_attn=True)
    except Exception as e:
        print(f"  ✗ ИСКЛЮЧЕНИЕ в forward: {e}")
        hook_mgr.remove()
        return
    hook_mgr.remove()

    if hook_mgr.first_nan:
        print(f"  ✗ Первый NaN/Inf в модуле:")
        for k, v in hook_mgr.first_nan.items():
            print(f"      {k}: {v}")
    else:
        print(f"  ✓ Forward pass чист")

    print(f"\n[4] Выход модели y:")
    ok_y = torch.isfinite(y).all()
    print(f"  {'✓' if ok_y else '✗ NaN/Inf'} shape={tuple(y.shape)} "
          f"min={float(y.min()):.4f} max={float(y.max()):.4f}")

    # 5.4 Loss
    print(f"\n[5] Loss:")
    try:
        loss = loss_fn(y, fake_targets)
        ok_loss = torch.isfinite(loss)
        print(f"  {'✓' if ok_loss else '✗ NaN/Inf'} loss={float(loss):.6f}")
    except Exception as e:
        print(f"  ✗ ИСКЛЮЧЕНИЕ в loss: {e}")
        return

    # 5.5 Diversity loss
    if hasattr(rnn, 'compute_diversity_loss') and extras:
        print(f"\n[6] Diversity loss:")
        try:
            div = rnn.compute_diversity_loss(extras)
            for k, v in div.items():
                ok = torch.isfinite(v)
                print(f"  {'✓' if ok else '✗ NaN/Inf'} {k}: {float(v):.6f}")
            total_loss = loss + div.get('total', torch.tensor(0.0))
        except Exception as e:
            print(f"  ✗ ИСКЛЮЧЕНИЕ в diversity_loss: {e}")
            total_loss = loss
    else:
        total_loss = loss

    # 5.6 Backward
    print(f"\n[7] Backward pass:")
    rnn.zero_grad()
    try:
        total_loss.backward()
        print(f"  ✓ backward завершён без исключений")
    except Exception as e:
        print(f"  ✗ ИСКЛЮЧЕНИЕ в backward: {e}")
        return

    # 5.7 Градиенты — подробно
    print(f"\n[8] Градиенты (только плохие):")
    bad_grads = check_grads(rnn)
    if bad_grads:
        for s in bad_grads:
            print(s)
    else:
        print("  ✓ Все градиенты конечны")

    # 5.8 Топ-10 наибольших градиентов (даже если finite)
    print(f"\n[9] Топ-10 параметров по норме градиента:")
    grad_norms = []
    for name, p in rnn.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            grad_norms.append((name, float(p.grad.norm().item())))
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    for name, norm in grad_norms[:10]:
        print(f"  {norm:10.4f}  {name}")

    # 5.9 Пошаговая диагностика grid step
    print(f"\n[10] Пошаговая диагностика _grid_step:")
    with torch.no_grad():
        x_emb = rnn.embedding(fake_tokens.view(-1))
        debug_grid_step(rnn, x_emb, rnn.init_state(bsz))

    print("\n" + "═" * 70)
    print("NaN DIAGNOSTICS END")
    print("═" * 70 + "\n")