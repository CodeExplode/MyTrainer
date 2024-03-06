A very simple Stable Diffusion 1.5 trainer, based on code from several other repositories and some of my own. Made for easier experimentation, and shared for people who wanted the embeddings inserter logic.

Uses hardcoded locations and references to my customized DataLoader which isn't included here. You will need to implement your own DataLoader to run this.

If the training exits due to an error before the first time the model backups are saved (at the end of the first epoch with the default settings), the "cache/model name/" folder will need to be deleted or the code will error out when trying to resume training.

requirements.txt is copied from OneTrainer and lightly pruned, and might install excess requirements which this code doesn't use. The install and start bat files are also copied from OneTrainer.

This repo is purely experimental, and contains very little error checking or adaptability from settings files etc. The textual inversion code has not been tested yet.