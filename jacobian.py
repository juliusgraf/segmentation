import torch
from model import UNet

model = UNet(in_channels=3, out_channels=1)

def jacobian_spectral_norm(y_in, x_hat, interpolation=True, training=False, max_it=5):

    if interpolation:
        eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).cuda()
        x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
        x = x.cuda()     ##to(self.device)
    else:
        x = y_in

    x.requires_grad_()
    x_hat=model(x)          ###x_hat, _ = self.forward(x, sigma)


    if y_in.shape != x_hat.shape:  # We pad with zeros when output of network (x_hat) and input (y_in) don't have same shape
      zer_pad = torch.zeros_like(x_hat)
      y_in = torch.cat((y_in, zer_pad[:,y_in.shape[1]:,...]), 1)

    y = 2. * x_hat - y_in  # Beware notation: y_in = input, x_hat = output network

    u = torch.randn_like(x)
    u = u / torch.norm(u, p=2)

    z_old = torch.zeros(u.shape[0])

    for it in range(max_it):  ##self.hparams.power_method_nb_step=50
        w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
        v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju
        v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

        z = torch.dot(u.flatten(), v.flatten()) / (torch.norm(u, p=2) ** 2)
        # z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

        if it > 0:
            rel_var = torch.norm(z - z_old)
            if rel_var < 1e-2:   #self.hparams.power_method_error_threshold=1e-2
                break
        z_old = z.clone()
        u = v / torch.norm(v, p=2)  # Modified

        if not training : ##modifier self.eval
            w.detach_()
            v.detach_()
            u.detach_()

    return z.view(-1)