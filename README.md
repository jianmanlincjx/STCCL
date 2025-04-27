# Learning Spatial-Temporal Coherent Correlationsfor Speech-Preserving Facial Expression Manipulation
## Overview
Speech-preserving facial expression manipulation (SPFEM) aims to modify facial emotions while meticulously maintaining the mouth animation associated with spoken content. Current works depend on inaccessible paired training samples for the person, where two aligned frames exhibit the same speech content yet differ in emotional expression, limiting the SPFEM applications in real-world scenarios. In this work, we discover that speakers who convey the same content with different emotions exhibit highly correlated local facial animations in both spatial and temporal spaces, providing valuable supervision for SPFEM. To capitalize on this insight, we propose a novel spatial-tempral coherent correlation learning (STCCL) algorithm, which models the aforementioned correlations as explicit metrics and integrates the metrics to supervise manipulating facial expression and meanwhile better preserving the facial animation of spoken contents. To this end, it first learns a spatial coherent correlation metric, ensuring that the visual correlations of adjacent local regions within an image linked to a specific emotion closely resemble those of corresponding regions in an image linked to a different emotion. Simultaneously, it develops a temporal coherent correlation metric, ensuring that the visual correlations of specific regions across adjacent image frames associated with one emotion are similar to those in the corresponding regions of frames associated with another emotion. Recognizing that visual correlations are not uniform across all regions, we have also crafted a correlation-aware adaptive strategy that prioritizes regions that present greater challenges. During SPFEM model training, we construct the spatial-temporal coherent correlation metric between corresponding local regions of the input and output image frames as addition loss to supervise the generation process. We conduct extensive experiments on variant datasets, and the results demonstrate the effectiveness of the proposed STCCL algorithm.

Here are some visual results of our approach:

![Visual Results](MEAD_RAVDESS.jpg)

Here are the quantitative results of integrating STCCL into other methods.
![Visual Results](MEAD.png)

![Visual Results](RAVDESS.png)

Here are some video results of our approach: video_result/*.mp4


## Train Pipeline

### Data Processing
Follow our previous work on data preprocessing at [ASCCL](https://github.com/jianmanlincjx/ASCCL).

### Training
Run `trainer/train_stccl.py` to obtain the checkpoints.

### Integration with Baseline
Follow our previous work for integration and evaluation using the steps from [ASCCL](https://github.com/jianmanlincjx/ASCCL) and [SSERD](https://github.com/ZH-Xu410/SSERD).

## Citation

If you use this work in your research, please cite the following:

```bibtex
@inproceedings{chen2024learning,
  title={Learning adaptive spatial coherent correlations for speech-preserving facial expression manipulation},
  author={Chen, Tianshui and Lin, Jianman and Yang, Zhijing and Qing, Chunmei and Lin, Liang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={7267--7276},
  year={2024}
}


