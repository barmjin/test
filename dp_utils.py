# dp_utils.py
from typing import Optional, Tuple
import torch
from opacus import PrivacyEngine

def attach_privacy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    target_delta: float = 1e-5,
    secure_mode: bool = False,  # شغّله للإنتاج لاحقًا
) -> Tuple[PrivacyEngine, Optional[float], torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
    """
    يضبط PrivacyEngine ويُرجع (pe, sample_rate, model, optimizer, private_loader)
    - نحتسب sample_rate قبل make_private لأن الـ loader الجديد قد لا يملك batch_size.
    """
    ds_len = len(data_loader.dataset)
    bs = getattr(data_loader, "batch_size", None)
    sample_rate: Optional[float] = (bs / ds_len) if (bs is not None and ds_len > 0) else None

    pe = PrivacyEngine(secure_mode=secure_mode)
    private_model, private_optimizer, private_loader = pe.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return pe, sample_rate, private_model, private_optimizer, private_loader


def get_epsilon(pe: PrivacyEngine, sample_rate: Optional[float], epochs: int, delta: float = 1e-5) -> Optional[float]:
    # محاسب Opacus يتتبّع المعدّل داخليًا؛ لا نحتاج sample_rate هنا، نتركه احتياطيًا للتوافق
    try:
        return float(pe.accountant.get_epsilon(delta=delta))
    except Exception:
        return None
