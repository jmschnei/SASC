# sc_common/trainer.py — shared train/eval loop with optional fp16 + wandb log_fn
import os
import torch
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, model, task, optimizer, scheduler, device, save_dir:str,
                 fp16:bool=False, log_fn=None):
        self.model, self.task = model, task
        self.opt, self.sched = optimizer, scheduler
        self.device, self.save_dir = device, save_dir
        self.fp16 = fp16 and torch.cuda.is_available()
        self.log_fn = log_fn  # e.g., lambda d: wandb.log(d)
        os.makedirs(save_dir, exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

    def _epoch(self, loader, train:bool):
        self.model.train() if train else self.model.eval()
        loss_sum, n_ex, agg = 0.0, 0, {}
        for batch in loader:
            with torch.set_grad_enabled(train):
                if train and self.fp16:
                    with torch.cuda.amp.autocast():
                        out = self.task.compute(self.model, batch, self.device)
                        loss = out['loss']
                else:
                    out = self.task.compute(self.model, batch, self.device)
                    loss = out['loss']

                if train:
                    self.opt.zero_grad(set_to_none=True)
                    if self.fp16:
                        self.scaler.scale(loss).backward()
                        clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.opt); self.scaler.update()
                    else:
                        loss.backward(); clip_grad_norm_(self.model.parameters(), 1.0); self.opt.step()
                    if self.sched: self.sched.step()

            bs = batch['input_ids'].size(0)
            loss_sum += float(loss.item()) * bs; n_ex += bs
            for k, v in out['metrics'].items():
                agg[k] = agg.get(k, 0.0) + float(v)

        logs = {'loss': loss_sum / max(1, n_ex)}
        for k, v in agg.items(): logs[k] = v / max(1, len(loader))
        return logs

    def fit(self, train_loader, eval_loader, epochs:int, save_cb=None):
        best = float('inf')
        for ep in range(1, epochs+1):
            tr = self._epoch(train_loader, True)
            dv = self._epoch(eval_loader,   False)
            print(f"[Epoch {ep}] train: {tr} | eval: {dv}")

            # log to W&B if provided
            if self.log_fn:
                payload = {f"train/{k}": v for k, v in tr.items()}
                payload.update({f"eval/{k}": v for k, v in dv.items()})
                payload['epoch'] = ep
                self.log_fn(payload)

            if dv['loss'] < best:
                best = dv['loss']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best.pt'))
                if save_cb: save_cb()
                print(f"✓ Saved best to {os.path.join(self.save_dir, 'best.pt')}")
        return best
