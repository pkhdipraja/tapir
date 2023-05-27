from pytorch_lightning import Callback


class LoggingCallback(Callback):
    def __init__(self, cfgs_dict):
        super(LoggingCallback, self).__init__()
        self.cfgs_dict = cfgs_dict

    def on_pretrain_routine_start(self, trainer, pl_module):
        trainer.logger.experiment.log_others(self.cfgs_dict)
