from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupReduceOnPlateau:
    def __init__(self, optimizer, warmup_steps, plateau_scheduler_kwargs, load_from_checkpoint=False):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups][:]

        if not load_from_checkpoint:
            for group in optimizer.param_groups:
                group['lr'] = 0.0

        self.plateau_scheduler = ReduceLROnPlateau(optimizer, **plateau_scheduler_kwargs)
        self.in_plateau_phase = False

    def step(self, metrics=None):

        if self.current_step <= self.warmup_steps:
            if metrics is None:
                # Warmup step
                self.current_step += 1
                warmup_factor = min(self.current_step / float(self.warmup_steps), 1.0)
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.base_lrs[i] * warmup_factor
            else:
                # In warmup, but metrics passed — skip
                return
        else:
            if not self.in_plateau_phase:
                self.in_plateau_phase = True
                print("Warmup complete. Switching to ReduceLROnPlateau.")

            if metrics is not None:
                # Plateau step
                self.current_step += 1
                self.plateau_scheduler.step(metrics)
            else:
                # Plateau phase, but no metric — skip
                return

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'base_lrs': self.base_lrs,
            'in_plateau_phase': self.in_plateau_phase,
            'plateau_scheduler': self.plateau_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.base_lrs = state_dict['base_lrs']
        self.in_plateau_phase = state_dict['in_plateau_phase']
        self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])
