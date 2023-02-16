import numpy as np
import os
from pathlib import Path
from scipy.stats import tukey_hsd
import matplotlib.pyplot as plt

def tukeys_test(args):
    path = args.common.dump_cc
    feats = os.listdir(path)
    modes = os.listdir(os.path.join(path, feats[0]))
    models = os.listdir(os.path.join(path, feats[0], modes[0]))
    print(feats, modes, models)
    
    for mode in modes:
        for model in models:
            features = {}
            for feat_name in feats:
                with open(os.path.join(path, feat_name, mode, model), 'rb') as f:
                    data = np.load(f)
                features[feat_name] = data.flatten()
                # print(features[feat_name].shape)
            data = list(features.values())
            res = tukey_hsd(*data)
            print(features.keys())
            # print(res)
            
            print(features.keys())
            print(res.pvalue.round(5))
            for v in ['']+list(features.keys()):
                print(v, end='|')
            print()
            for idx, line in enumerate(res.pvalue):
                print(list(features.keys())[idx], end='|')
                for v in line:
                    print(round(v,4), end='|')
                print()
            # exit()
            pvalue = res.pvalue.round(5)
            pvalue_thres = (pvalue >= 0.05).astype(np.int16)
            print(pvalue_thres)
            plt.imshow(pvalue_thres)
            plt.xticks([i for i in range(len(features.keys()))], list(features.keys()), rotation=45)
            plt.yticks([i for i in range(len(features.keys()))], list(features.keys()), rotation=0)
            plt.colorbar()
            plt.title('1=p>=0.05, 0=p<0.05')
            plt.savefig('tmct')
            plt.clf()
            plt.imshow(pvalue)
            names = list(features.keys())
            names = [' ' .join(l.upper().split('_')) for l in names]
            print(names)
            plt.xticks([i for i in range(len(features.keys()))], names, rotation=270)
            plt.yticks([i for i in range(len(features.keys()))], names, rotation=0)
            plt.colorbar()
            plt.title('p values')
            plt.tight_layout()
            plt.savefig('tmct_raw')
            # for j in range(12):
            #     data = [features[f][:,j] for f in features]
            #     res = tukey_hsd(*data)
            #     print(features.keys())
            #     print(res.pvalue.round(5))
