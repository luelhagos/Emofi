import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    MelStyleEncoder,
    PhonemeEncoder,
    MelDecoder,
    VarianceAdaptor,
    PhonemeDiscriminator,
    StyleDiscriminator,
)
from .postfilter import PostfilterGenerator, MelDiscriminator
from utils.tools import get_mask_from_lengths


class StyleSpeech(nn.Module):
    """ StyleSpeech """

    def __init__(self, preprocess_config, model_config):
        super(StyleSpeech, self).__init__()
        self.model_config = model_config

        self.mel_style_encoder = MelStyleEncoder(preprocess_config, model_config)
        self.phoneme_encoder = PhonemeEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.mel_decoder = MelDecoder(model_config)
        self.phoneme_linear = nn.Linear(
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_hidden"],
        )
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.D_t = PhonemeDiscriminator(preprocess_config, model_config)
        self.D_s = StyleDiscriminator(preprocess_config, model_config)

        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            ),
            "r",
        ) as f:
            n_speaker = len(json.load(f))
        self.style_prototype = nn.Embedding(
            n_speaker,
            model_config["melencoder"]["encoder_hidden"],
        )

        # Project the frozen face encoder embedding (768-d) to the style vector
        # dim (encoder_hidden = 128). Used instead of mel_style_encoder when
        # face_embs are provided to forward().
        self.face_to_style = nn.Linear(
            768,
            model_config["melencoder"]["encoder_hidden"],
        )

        # Emotion-classification auxiliary head on the style vector (trained with
        # a CE loss alongside the main objective).
        self.emotion_aux_head = nn.Linear(
            model_config["melencoder"]["encoder_hidden"],
            6,  # MEAD's 6 emotions: angry/fear/happy/surprised/sad/neutral
        )

        # Intensity-classification auxiliary head on the same style vector
        # (trained with a CE loss for the three intensity levels).
        self.intensity_aux_head = nn.Linear(
            model_config["melencoder"]["encoder_hidden"],
            3,  # level1 / level2 / level3
        )

        # Three-input factorization. face_to_style -> emotion axis of the style
        # vector; the speaker channel -> speaker axis; combine both before SALN.
        # Three combine mechanisms supported:
        #   - "sum"           : style = face_style + ref_style (no extra params)
        #   - "gated"         : style = a * face + (1-a) * ref ; a is a learnable
        #                       scalar passed through sigmoid for [0,1] range
        #   - "concat_linear" : style = Linear(2H, H)([face, ref])
        # Only used when forward(face_embs=..., combine_mode=<mode>) is called.
        H = model_config["melencoder"]["encoder_hidden"]
        self.combine_gate   = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
        self.combine_linear = nn.Linear(2 * H, H)

        # ECAPA speaker channel. Project a pretrained 192-d ECAPA-TDNN
        # speaker-verification embedding to the style dim -- a finer, largely
        # emotion-invariant alternative to mel_style_encoder for the speaker
        # axis. Used when forward(spk_embs=...) is passed.
        self.spk_to_style = nn.Linear(192, H)

        # Intensity postfilter (model/postfilter.py): a residual refiner on the
        # decoder mel, trained on the frozen backbone. Enabled via the
        # model config's `postfilter` block (see config/MEAD/model_intensity.yaml).
        self.postfilter_g = self.postfilter_d = None
        pcfg = model_config.get("postfilter", {}) if model_config else {}
        if pcfg.get("enabled", False):
            n_mel_pf = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
            self.postfilter_g = PostfilterGenerator(
                n_mel=n_mel_pf, style_dim=H,
                hidden=pcfg.get("hidden", 256), n_layers=pcfg.get("n_layers", 6),
                level_cond=pcfg.get("level_cond", False))
            self.postfilter_d = MelDiscriminator(
                n_mel=n_mel_pf, hidden=pcfg.get("disc_hidden", 256),
                n_scales=pcfg.get("n_scales", 3))

    def compute_emotion_aux(self, face_embs):
        """Return aux emotion logits (B, 6) from face_embs. Called from
        train.py to compute an auxiliary CE loss on the style vector.
        Mirrors the projection used in forward() so the same Linear is
        trained against both objectives."""
        return self.emotion_aux_head(self.face_to_style(face_embs))

    def compute_intensity_aux(self, face_embs):
        """Return aux intensity logits (B, 3) from face_embs. Same idea as
        compute_emotion_aux but for the three MEAD intensity levels. Also used
        at inference to predict the intensity level from the face alone."""
        return self.intensity_aux_head(self.face_to_style(face_embs))

    def compute_style_vector(self, face_embs, spk_embs=None, ref_mels=None,
                             mels=None, mel_masks=None, combine_mode=None):
        """Recompute the style vector (B, 1, H) the same way forward() does, so
        the postfilter can be conditioned on it outside the main forward."""
        face_style = self.face_to_style(face_embs).unsqueeze(1)
        if combine_mode is None:
            return face_style
        if spk_embs is not None:
            ref_style = self.spk_to_style(spk_embs).unsqueeze(1)
        else:
            ref = ref_mels if ref_mels is not None else mels
            ref_masks = (ref.abs().sum(dim=-1) == 0) if ref is not mels else mel_masks
            ref_style = self.mel_style_encoder(ref, ref_masks)
        if combine_mode == "sum":
            return face_style + ref_style
        if combine_mode == "gated":
            a = torch.sigmoid(self.combine_gate)
            return a * face_style + (1.0 - a) * ref_style
        if combine_mode == "concat_linear":
            return self.combine_linear(torch.cat([face_style, ref_style], dim=-1))
        raise ValueError(f"Unknown combine_mode: {combine_mode!r}")

    # ---- intensity postfilter ------------------------------------------------

    def postfilter_refine(self, mu, style_vector, level=None, level_weights=None):
        """Inference: refine mu -> mu + Delta. mu: (B, T, n_mel); style_vector:
        (B, 1, H); level: (B,) in {0, 1, 2} or None; level_weights: (B, 3) soft
        mixture over the level embeddings (continuous intensity control)."""
        refined = self.postfilter_g(mu.transpose(1, 2), style_vector, level,
                                    level_weights=level_weights)
        return refined.transpose(1, 2)

    def _pf_inputs(self, mu, gt_mel, mel_masks):
        """Shared (mu_t, gt_t, valid) in (B, n_mel, T); valid (B, 1, T) 1=keep."""
        mu_t = mu.transpose(1, 2)
        gt_t = gt_mel[:, : mu.shape[1], :].transpose(1, 2)
        valid = (~mel_masks[:, : mu.shape[1]]).unsqueeze(1).float()
        return mu_t, gt_t, valid

    def postfilter_d_loss(self, mu, gt_mel, style_vector, mel_masks, levels=None):
        """LSGAN discriminator: real (gt) -> 1, refined -> 0, summed over scales."""
        mu_t, gt_t, valid = self._pf_inputs(mu, gt_mel, mel_masks)
        with torch.no_grad():
            refined = self.postfilter_g(mu_t, style_vector, levels) * valid
        gt_t = gt_t * valid
        loss = 0.0
        for (lr, _), (lf, _) in zip(self.postfilter_d(gt_t), self.postfilter_d(refined)):
            loss = loss + ((lr - 1.0) ** 2).mean() + (lf ** 2).mean()
        return loss

    def postfilter_g_loss(self, mu, gt_mel, style_vector, levels, natural_std, mel_masks,
                          lambda_fm=2.0, lambda_gv=1.0, lambda_con=0.5):
        """Generator: LSGAN adversarial + discriminator feature matching +
        per-level global-variance + content anchor (to mu). levels: (B,) in
        {0, 1, 2}; natural_std: (3, n_mel) per-level target."""
        mu_t, gt_t, valid = self._pf_inputs(mu, gt_mel, mel_masks)
        refined = self.postfilter_g(mu_t, style_vector, levels)
        refined_m, gt_m = refined * valid, gt_t * valid
        d_ref, d_gt = self.postfilter_d(refined_m), self.postfilter_d(gt_m)
        l_adv = sum(((lf - 1.0) ** 2).mean() for lf, _ in d_ref)
        l_fm = 0.0
        for (_, fr), (_, fg) in zip(d_ref, d_gt):
            for a, b in zip(fr, fg):
                l_fm = l_fm + F.l1_loss(a, b.detach())
        # per-level GV: rendered per-bin std over valid frames vs natural target
        n = valid.sum(dim=2).clamp(min=1.0)                        # (B, 1)
        mean = (refined * valid).sum(dim=2) / n                    # (B, n_mel)
        var = (((refined - mean.unsqueeze(2)) ** 2) * valid).sum(dim=2) / n
        std = torch.sqrt(var + 1e-8)                               # (B, n_mel)
        l_gv = F.l1_loss(std, natural_std[levels])
        l_con = F.l1_loss(refined_m, mu_t * valid)                 # content anchor
        total = l_adv + lambda_fm * l_fm + lambda_gv * l_gv + lambda_con * l_con
        return total, {"adv": l_adv.item(), "fm": float(l_fm) if isinstance(l_fm, float)
                       else l_fm.item(), "gv": l_gv.item(), "con": l_con.item()}

    def G(
            self,
            style_vector,
            texts,
            src_masks,  
            mel_masks,
            max_mel_len,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
        ):
        output = self.phoneme_encoder(texts, style_vector, src_masks)
        output = self.phoneme_linear(output)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.mel_decoder(output, style_vector, mel_masks)
        output = self.mel_linear(output)

        return (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        )

    def forward(
        self,
        _,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        face_embs=None,
        ref_mels=None,
        spk_embs=None,
        combine_mode=None,
        zero_face=False,
        zero_ref=False,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        if face_embs is not None and combine_mode is not None:
            # Three-input factorization -- face = emotion, speaker channel is
            # EITHER a mel-style ref (mel_style_encoder) OR a pretrained ECAPA
            # speaker embedding (spk_to_style). The face half is identical; only
            # the speaker component differs.
            face_style = self.face_to_style(face_embs).unsqueeze(1)   # (B, 1, H)
            if spk_embs is not None:
                # ECAPA speaker embedding (192-d, L2-normalized) -> style.
                ref_style = self.spk_to_style(spk_embs).unsqueeze(1)  # (B, 1, H)
            else:
                # Mel-style encoder on a reference mel.
                ref = ref_mels if ref_mels is not None else mels
                if ref is mels:
                    ref_masks = mel_masks
                else:
                    ref_masks = (ref.abs().sum(dim=-1) == 0)  # True = padding
                ref_style = self.mel_style_encoder(ref, ref_masks)   # (B, 1, H)
            # Ablation (input-necessity test): zero one style component.
            if zero_face:
                face_style = torch.zeros_like(face_style)
            if zero_ref:
                ref_style = torch.zeros_like(ref_style)
            if combine_mode == "sum":
                style_vector = face_style + ref_style
            elif combine_mode == "gated":
                alpha = torch.sigmoid(self.combine_gate)
                style_vector = alpha * face_style + (1.0 - alpha) * ref_style
            elif combine_mode == "concat_linear":
                style_vector = self.combine_linear(
                    torch.cat([face_style, ref_style], dim=-1)
                )
            else:
                raise ValueError(f"Unknown combine_mode: {combine_mode!r}")
        elif face_embs is not None:
            # Face only
            style_vector = self.face_to_style(face_embs).unsqueeze(1)
        else:
            # Original mel-style only
            style_vector = self.mel_style_encoder(mels, mel_masks)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.G(
            style_vector,
            texts,
            src_masks,  
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        return (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    def meta_learner_1(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        raw_quary_texts=None,
        quary_texts=None,
        quary_src_lens=None,
        max_quary_src_len=None,
        quary_d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        quary_mel_lens = quary_d_targets.sum(dim=-1)
        max_quary_mel_len = max(quary_mel_lens).item()
        quary_src_masks = get_mask_from_lengths(quary_src_lens, max_quary_src_len)
        quary_mel_masks = get_mask_from_lengths(quary_mel_lens, max_quary_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        (
            output,
            _,
            _,
            _,
            d_rounded_adv,
            mel_lens_adv,
            mel_masks_adv,
        ) = self.G(
            style_vector,
            quary_texts,
            quary_src_masks,
            quary_mel_masks,
            max_quary_mel_len,
            None,
            None,
            None,
            p_control,
            e_control,
            d_control,
        )

        D_s = self.D_s(self.style_prototype, speakers, output, mel_masks_adv)

        quary_texts = self.phoneme_encoder.src_word_emb(quary_texts)
        D_t = self.D_t(self.variance_adaptor.upsample, quary_texts, output, max(mel_lens_adv).item(), mel_masks_adv, d_rounded_adv)

        (
            G,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.G(
            style_vector,
            texts,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        return (
            D_s,
            D_t,
            G,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    def meta_learner_2(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        raw_quary_texts=None,
        quary_texts=None,
        quary_src_lens=None,
        max_quary_src_len=None,
        quary_d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        quary_mel_lens = quary_d_targets.sum(dim=-1)
        max_quary_mel_len = max(quary_mel_lens).item()
        quary_src_masks = get_mask_from_lengths(quary_src_lens, max_quary_src_len)
        quary_mel_masks = get_mask_from_lengths(quary_mel_lens, max_quary_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        (
            output,
            _,
            _,
            _,
            d_rounded_adv,
            mel_lens_adv,
            mel_masks_adv,
        ) = self.G(
            style_vector,
            quary_texts,
            quary_src_masks,
            quary_mel_masks,
            max_quary_mel_len,
            None,
            None,
            None,
            p_control,
            e_control,
            d_control,
        )

        texts = self.phoneme_encoder.src_word_emb(texts)
        D_t_s = self.D_t(self.variance_adaptor.upsample, texts, mels, max_mel_len, mel_masks, d_targets)

        quary_texts = self.phoneme_encoder.src_word_emb(quary_texts)
        D_t_q = self.D_t(self.variance_adaptor.upsample, quary_texts, output, max(mel_lens_adv).item(), mel_masks_adv, d_rounded_adv)

        D_s_s = self.D_s(self.style_prototype, speakers, mels, mel_masks)
        D_s_q = self.D_s(self.style_prototype, speakers, output, mel_masks_adv)

        # Get Style Logit
        w = style_vector.squeeze() # [B, H]
        style_logit = torch.matmul(w, self.style_prototype.weight.contiguous().transpose(0, 1)) # [B, K]

        return D_t_s, D_t_q, D_s_s, D_s_q, style_logit

