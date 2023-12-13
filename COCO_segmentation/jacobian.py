import torch
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)

def jacobian_spectral_norm(y_in, x_hat, interpolation=True, training=False, max_it=5):
    if interpolation:
        eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(device)
        x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
        x = x.to(device)
    else:
        x = y_in

    x.requires_grad_()
    x_hat = model(x)

    if y_in.shape != x_hat.shape:
        zer_pad = torch.zeros_like(x_hat)
        y_in = torch.cat((y_in, zer_pad[:, y_in.shape[1]:, ...]), 1)

    y = x_hat

    u = torch.randn_like(x).to(device)
    u = u / torch.norm(u, p=2)

    z_old = torch.zeros(u.shape[0]).to(device)

    for it in range(max_it):
        w = torch.ones_like(y, requires_grad=True)
        v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]
        v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)

        z = torch.dot(u.flatten(), v.flatten()) / (torch.norm(u, p=2) ** 2)

        if it > 0:
            rel_var = torch.norm(z - z_old)
            if rel_var < 1e-2:
                break
        z_old = z.clone()
        u = v / torch.norm(v, p=2)

        if not training:
            w.detach_()
            v.detach_()
            u.detach_()

    return z.view(-1)