# ssl_aai
codes for the work accepted at ICASSP 2023 - "Improved acoustic-to-articulatory inversion using representations from pretrained self-supervised learning models", preprint - https://arxiv.org/abs/2210.16871

Code description -
- Change experiment parameters in config/hparams.yaml
- Run train.py
- Encoder only transformer model is used, the core implementation taken from NVIDIA-fastpitch
- self supervised features are extracted using s3prl toolkit
- Correlation coefficient is computed to evaluate the models

For any queries, contact sathvikudupa66@gmail.com
