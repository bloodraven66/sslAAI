import wandb
import matplotlib.pyplot as plt
import seaborn as sns

class WandbLogger():
    def __init__(self, config):
        self.run = wandb.init(
                            name=config.logging.run_name,
                            project=config.logging.project_name,
                            config=config,
                            notes=config.logging.notes,
                            tags=config.logging.tags)

    def log(self, dct):
        wandb.log(dct)


    
    def plot_ema(self, ema, out, lens, num=2):
        for i in range(num):
            ema_ = ema[i][:lens[i].int()].detach().cpu().numpy()
            out_ = out[i][:lens[i].int()].detach().cpu().numpy()
            fig, ax = plt.subplots(6, 2)
            for j in range(6):
                ax[j][0].plot(ema_[:, 2*j], color='black')
                ax[j][0].plot(out_[:, 2*j], color='red', linestyle=":")
                ax[j][1].plot(ema_[:, 2*j+1], color='black')
                ax[j][1].plot(out_[:, 2*j+1], color='red', linestyle=":")
            plt.tight_layout()
            plt.savefig(f'plots/{i}')
            plt.clf()
        # exit()
    def summary(self, dct):
        for key in dct:
            wandb.run.summary[key] = dct[key]

    def log_audio(Self, aud, name="val", sample_rate=22050):
        wandb.log({name: wandb.Audio(aud,  sample_rate=sample_rate)})

    def end_run(self):
        self.run.finish()
