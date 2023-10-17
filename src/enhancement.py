#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from src.utils import LinearScheduler, calc_metrics
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel
from torchaudio import load
from sgmse.util.other import pad_spec


class UDiffSE:
    def __init__(
        self,
        ckpt_path="data/checkpoints/diffusion_gen_nonlinear_transform.ckpt",
        num_E=30,
        transform_type="exponent",
        delta=1e-10,
        eps=0.03,
        snr=0.5,
        sr=16000,
        verbose=False,
        device="cuda",
    ):
        """
        Unsupervised Diffusion-Based Speech Enhancement (UDiffSE) algorithm.

        Args:
            ckpt_path: Path to the pre-trained diffusion model.
            num_E: Number of iterations for the E step (reverse diffusion process).
            verbose: Whether to print progress information.
        """

        self.snr = snr
        self.sr = sr
        self.delta = delta
        self.num_E = num_E

        self.verbose = verbose
        self.device = device
        self.scheduler = LinearScheduler(N=num_E, eps=eps)
        self.sde = OUVESDE(theta=1.5, sigma_min=0.05, sigma_max=0.5, N=num_E)

        # ==== Prior model ====
        self.model = ScoreModel.load_from_checkpoint(
            ckpt_path, base_dir="", batch_size=1, num_workers=0, kwargs=dict(gpu=False)
        )
        self.model.data_module.transform_type = transform_type
        self.model.eval(no_ema=False)
        self.model.to(self.device)

    def load_data(self, file_path):
        """
        Load speech data and compute spectrogram.
        """
        x, sr = load(file_path)
        assert sr == self.sr
        self.T_orig = x.size(1)

        X = pad_spec(
            torch.unsqueeze(self.model._forward_transform(self.model._stft(x)), 0)
        ).to(self.device)

        return x, X

    def to_audio(self, specto):
        specto = specto * self.NF
        return self.model.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1)

    def predictor_corrector(self, St, t, laststep, dt):
        with torch.no_grad():
            # Corrector
            score = self.model.forward(St, t)
            std = self.sde.marginal_prob(St, t)[1]
            step_size = (self.snr * std) ** 2
            z = torch.randn_like(St)
            St = (
                St
                + step_size[:, None, None, None] * score
                + torch.sqrt(step_size * 2)[:, None, None, None] * z
            )

            # Predictor
            f, g = self.sde.sde(St, t)
            score = self.model.forward(St, t)
            z = (
                torch.zeros_like(St) if laststep else torch.randn_like(St)
            )  # if not laststep else torch.zeros_like(St)
            St = (
                St
                - f * dt
                + (g**2)[:, None, None, None] * score * dt
                + g[:, None, None, None] * torch.sqrt(dt) * z
            )
            torch.cuda.empty_cache()

        return St, std, score, g

    def likelihood_update(self, St, t, std, dt):
        """
        Pseudo-likelihood update.
        """
        with torch.no_grad():
            theta = self.sde.theta
            mu_t = torch.exp(-theta * t)[:, None, None, None]
            _, g = self.sde.sde(St, t)

            difference = self.X - St / mu_t
            nppls = (
                (1 / mu_t)
                * difference
                / ((std[:, None, None, None] / mu_t) ** 2 + self.Vt)
            ).type(torch.complex64)

            weight = self.lmbd * (g**2)[:, None, None, None]
            St = St + weight * nppls * dt
            return St

    def prior_sampler(self):
        """
        Prior sampling algorithm to unconditionally generate a clean speech signal.
        """
        timesteps = self.scheduler.timesteps()
        self.NF = 1
        self.T_orig = 80000

        # Set the very first sample at t=1
        St = torch.randn(
            1, 1, 256, 640, dtype=torch.cfloat, device=self.device
        ) * self.sde._std(torch.ones(1, device=self.device))

        # Discretised time-step
        dt = torch.tensor(1 / self.num_E, device=self.device)

        # Sampling iterations
        for i in tqdm(range(0, self.num_E)):
            t = torch.tensor([timesteps[i]], device=self.device)
            St, _, _, _ = self.predictor_corrector(
                St=St,
                t=t,
                laststep=i == (self.num_E - 1),
                dt=dt,
            )

        st = self.to_audio(St)

        return st, St

    def posterior_sampler(self, startstep=0, skip_EM1=False):
        """
        Posterior sampler algorithm that functions as the E-step for the EM process of UDiffSE.
        """
        timesteps = self.scheduler.timesteps()

        # Set the very first sample at t=1
        St = (
            torch.randn_like(self.X) * self.sde._std(torch.ones(1, device=self.device))
            + self.X
        )

        # Discretised time-step
        dt = torch.tensor(1 / self.num_E, device=self.device)

        if self.verbose:
            range_i = tqdm(range(startstep, self.num_E))
        else:
            range_i = range(startstep, self.num_E)

        for i in range_i:
            # Predictor-Corrector iteration
            t = torch.tensor([timesteps[i]], device=self.device).repeat(self.nbatch)
            St, std, _, _ = self.predictor_corrector(
                St=St,
                t=t,
                laststep=i == (self.num_E - 1),
                dt=dt,
            )

            # Likelihood term
            if i % self.project_every_k_steps == 0 and not skip_EM1:
                St = self.likelihood_update(
                    St=St,
                    t=t,
                    std=std,
                    dt=dt,
                )

        return St

    def parameter_update(self, X_init_st, W, H):
        Vm = (X_init_st).abs().pow(2).mean(0).unsqueeze(0)
        # temporary
        V = W @ H

        # Update W
        num = (Vm * V.pow(-2)) @ H.permute(0, 1, 3, 2)
        den = V.pow(-1) @ H.permute(0, 1, 3, 2)
        W = W * (num / den)
        W = torch.maximum(W, torch.tensor([self.delta], device=self.device))

        # Update V
        V = W @ H

        # Update H
        num = W.permute(0, 1, 3, 2) @ (Vm * V.pow(-2))  # transpose
        den = W.permute(0, 1, 3, 2) @ V.pow(-1)
        H = H * (num / den)
        H = torch.maximum(H, torch.tensor([self.delta], device=self.device))

        # Normalise
        norm_factor = torch.sum(W.abs(), axis=2)
        W = W / torch.unsqueeze(norm_factor, 2)
        H = H * torch.unsqueeze(norm_factor, 3)

        return W, H

    def run(
        self,
        mix_file,
        clean_file=None,
        num_EM=5,
        lmbd=1.5,
        nbatch=2,
        nmf_rank=4,
        project_every_k_steps=2,
    ):
        self.lmbd = lmbd
        self.project_every_k_steps = project_every_k_steps
        self.nbatch = nbatch

        x, X = self.load_data(mix_file)
        self.x = x
        self.NF = X.abs().max()
        X = X / self.NF

        if self.verbose and clean_file != None:
            s_ref, S_ref = self.load_data(clean_file)
            self.s_ref = s_ref
            self.S_ref = S_ref
            s_ref = s_ref.numpy().reshape(-1)
            x = x.numpy().reshape(-1)
            metrix = calc_metrics(s_ref, x)
            print(
                f"Input PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f}",
                end="\r",
            )
            print("")

        self.X = X.repeat(self.nbatch, 1, 1, 1)
        metrix = {"pesq": 0.0, "si_sdr": 0.0, "estoi": 0.0}

        # Initialise W and H (NMF matrices)
        _, _, T, F = X.shape
        Wt = torch.rand(T, nmf_rank, device=self.device).clamp_(min=self.delta)[
            None, None, :, :
        ]
        Ht = torch.rand(nmf_rank, F, device=self.device).clamp_(min=self.delta)[
            None, None, :, :
        ]
        self.Vt = Wt @ Ht

        # EM algorithm
        for j in range(num_EM):
            # E-step (posterior sampler)
            if j == 0:  # Don't do likelihood update at the 1st EM iteration
                St = self.posterior_sampler(
                    skip_EM1=True,
                )
            else:
                St = self.posterior_sampler(
                    skip_EM1=False,
                )

            # M-step (W&H updates)
            Wt, Ht = self.parameter_update(self.X - St, Wt, Ht)
            self.Vt = Wt @ Ht

            St = St.mean(0)
            st = self.to_audio(St).numpy().reshape(-1)
            if self.verbose and clean_file != None:
                metrix = calc_metrics(s_ref, st)
                print(
                    f"{j}/{num_EM} PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f} --- ESTOI: {metrix['estoi']:.4f}",
                    end="\r",
                )
                print("")

        return st, St
