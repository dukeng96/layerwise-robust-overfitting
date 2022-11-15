# Review 1
We sincerely thank you for the comprehensive comments on our paper and please find the answers to your questions below.

**Q1:** Lack of comparison with SOTA methods.

**A1:** Below are the robust test performance comparisons with some of the existing SOTA methods under PreAct-ResNet18 architecture, $\ell_{\infty}$ threat model and CIFAR-10 dataset

|Method|Best|Last|Diff|
| :-----| ----: | :----: |:----: |
|AT|52.31|44.45|7.86
|AWP|55.39|54.73|0.66
|SWA|52.14|51.53|0.61
|GAIRAT|57.81|53.76|4.05
|RAT_LR|51.57|49.07|2.5
|RAT_WP|54.85|53.98|0.87

Reference \
AWP: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf \
SWA: https://openreview.net/pdf?id=qZzy5urZw9 \
GAIRAT: https://arxiv.org/pdf/2010.01736.pdf 


**Q2:** Further explanation for AT-fix-param-[XXX].

**A2:** Fixing parameter in this case means the parameters are fixed as random initialization as you expected. And yes, we still update the fc layer. We only isolate different ResNet blocks and treated them as different layers.

**Q3:** Code release for reproducibility

**A2:** Codes are available at https://github.com/dukeng96/layerwise-robust-overfitting

# Review 2
We sincerely thank you for the comprehensive comments on our paper and please find the answers to your questions below.

**Q1:** The contributions and novelty are fairly limited.

**A1:** Our main contribution is the discovery of different properties of each network layer towards the robust overfitting phenomenon. Our methods serve to highlight that the latter layers are more sensitive to robust overfitting, and thus constraining its optimization can significantly reduce overfitting. There could be better methods of achieving this goal, instead of training with a fixed learning rate or AWP for latter layers. It is still an open question to design an optimal method that utilize this discovery, and we hope future researchs can achieve this.  

**Q2:** The deeper layers/blocks are known to have many more parameters, thus it is not entirely surprising that the latter layers contribute more to overfitting

**A2:** Assume that robust overfitting is related to the number of parameters, then the experiments AT_fix_lr_[1,2,4] (figure 2) and AT_awp_[1,2,4] (figure 3) should practically reduce robust overfitting even slightly, as in the experiments AT_fix_lr_[3,4] (figure 2) and AT_awp_[3,4] (figure 3). Since the number of parameters are almost the same between these two cases, but the effects on robust overfitting are quite divergent, we believe that the number of parameters is not a significant cause of robust overfitting. 

**Q3:** Lack of clean accuracy report

**A3:** Below are the best natural accuracy report under PreAct-ResNet18 architecture, $\ell_{\infty}$ threat model. We generally observe that our methods do not introduce trade-off in natural accuracy.    

|Method|CIFAR-10|CIFAR-100|SVHN|
| :-----| ----: | :----: |:----: |
|AT|85.28|56.08|91.72
|AWP|81.97|54.94|93.71
|RAT_LR|81.59|54.36|91.77
|RAT_WP|83.78|56.00|94.41

# Review 3
We sincerely thank you for the comprehensive comments on our paper and please find the answers to your questions below.

**Q1:** Deep layers are overparameterized and can be simply reduced to mitigate the robust overfitting.

**A1:** Assume that robust overfitting is related to the number of parameters, then the experiments AT_fix_lr_[1,2,4] (figure 2) and AT_awp_[1,2,4] (figure 3) should practically reduce robust overfitting even slightly, as in the experiments AT_fix_lr_[3,4] (figure 2) and AT_awp_[3,4] (figure 3). Since the number of parameters are almost the same between these two cases, but the effects on robust overfitting are quite divergent, we believe that the number of parameters is not a significant cause of robust overfitting.

**Q2:** Incomplete robustness evaluation with different AT methods

**A2:** Below are performance report for TRADES and MART, under PreAct-ResNet18 architecture, $\ell_{\infty}$ threat model and CIFAR10 dataset. In general, we find that TRADES or MART have negligible impact on robust overfitting. They can increase the robust accuracy performance slightly when applied to our methods or AWP.

|Method|Best|Last|Diff|
| :-----| ----: | :----: |:----: |
|AT|52.31|44.45|7.86
|AT_AWP|55.39|54.73|0.66
|AT_RAT_LR|51.57|49.07|2.5
|AT_RAT_WP|54.85|53.98|0.87
|TRADES|52.31|44.45|7.86
|TRADES_AWP|55.62|54.96|0.72
|TRADES_RAT_LR|51.63|49.05|2.58
|TRADES_RAT_WP|55.12|54.18|0.94
|MART|55.57|47.65|7.92
|MART_AWP|57.74|57.05|0.69
|MART_RAT_LR|52.98|51.12|1.85
|MART_RAT_WP|57.32|56.35|0.97

**Q3:** Why latter layers are so special?

**A3:** One intuition is that DNNs are often optimized with backpropagation using SGD, supervisory signals will gradually propagate through the whole network from latter layers to former layers. In this sense, the latter layers might be more sensitive to the supervisory signals coming from the adversarial loss compared to their former counterparts. In this work https://arxiv.org/pdf/2106.15853.pdf, the authors also discover that the latter layers in a DNN are
much more sensitive to label noise, while their former counterparts are quite robust. If we somehow consider adversarial data as "input noise", the same dynamics from label noise can apply.

# Review 4
We sincerely thank you for the comprehensive comments on our paper and please find the answers to your questions below.

**Q1:** Fixing different blocks will result in the models with different number of effective parameters so that they will have different capacity

**A1:** Assume that robust overfitting is related to the number of parameters, then the experiments AT_fix_lr_[1,2,4] (figure 2) and AT_awp_[1,2,4] (figure 3) should practically reduce robust overfitting even slightly, as in the experiments AT_fix_lr_[3,4] (figure 2) and AT_awp_[3,4] (figure 3). Since the number of parameters are almost the same between these two cases, but the effects on robust overfitting are quite divergent, we believe that the number of parameters does not have a strong relation to robust overfitting.

**Q2:** Lack of novelty. Performance not as good as AWP

**A2:** Our main contribution is the discovery of different properties of each network layer towards the robust overfitting phenomenon. Our methods serve to highlight that the latter layers are more sensitive to robust overfitting, and thus constraining its optimization can significantly reduce overfitting. There could be better methods of achieving this goal, instead of training with a fixed learning rate or AWP for latter layers. It is still an open question to design an optimal method that utilize this discovery, and we hope future researchs can achieve this.

**Q3:** Comparison with AWP

**A3:** Below are the performance comparisons with AWP under PreAct-ResNet18 architecture, $\ell_{\infty}$ threat model. In general, RAT_LR performance is outperformed by AWP by a good margin, and RAT_WP's performance is slightly lower than AWP. However, our main contribution is not about AT performance gain compared to other SOTA methods. We aim to highlight the layer-wise dynamics of robust overfitting and propose a prototype to reduce robust overfitting based on it. Although AWP is considered the current SOTA method, our experiments show that AWP if not applied to the latter layers will not be effective in mitigating robust overfitting. 

- CIFAR10

|Method|Best|Last|Diff|
| :-----| ----: | :----: |:----: |
|AWP|55.39|54.73|0.66
|RAT_LR|51.57|49.07|2.5
|RAT_WP|54.85|53.98|0.87

- CIFAR100

|Method|Best|Last|Diff|
| :-----| ----: | :----: |:----: |
|AWP|30.71|30.28|0.66
|RAT_LR|26.57|26.18|0.39
|RAT_WP|30.91|30.42|0.49

- SVHN

|Method|Best|Last|Diff|
| :-----| ----: | :----: |:----: |
|AWP|59.12|55.87|3.25
|RAT_LR|53.32|43.41|9.92
|RAT_WP|57.91|54.32|3.58

**Q4:** How are the last fully connected layers treated in the experiments? Are they fixed?

**A4:** We still update the FC layer. We only isolate different ResNet blocks and treated them as different layers.




















