import torch
import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, pred, truth):
        (
            mel_preds,
            postnet_mel_preds,
            pitch_preds,
            energy_preds,
            log_duration_preds,
            src_masks,
            mel_masks,
            _
        ) = pred

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            mel_targets,
            energy_targets,
            pitch_targets,
            duration_targets,
        ) = truth      

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        log_duration_targets = torch.log(duration_targets.float() + 1)

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        # mask
        pitch_preds = pitch_preds.masked_select(src_masks)
        pitch_targets = pitch_targets.masked_select(src_masks)

        energy_preds = energy_preds.masked_select(src_masks)
        energy_targets = energy_targets.masked_select(src_masks)

        log_duration_preds = log_duration_preds.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_preds = mel_preds.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_preds = postnet_mel_preds.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # loss
        mel_loss = self.mae_loss(mel_preds, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_preds, mel_targets)
        pitch_loss = self.mse_loss(pitch_preds, pitch_targets)
        energy_loss = self.mse_loss(energy_preds, energy_targets)
        duration_loss = self.mse_loss(log_duration_preds, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            energy_loss,
            pitch_loss,
            duration_loss,
        )
