# Frame-Level Multi-Label Playing Technique Detection Using Multi-Scale Network and Self-Attention Mechanism
## Dichucheng Li, Mingjin Che, Wenwu Meng, Yulun Wu, Yi Yu, Fan Xia, Wei Li


<!-- ### Abstract of the paper
```
Instrument playing technique (IPT) is a key element of musical presentation. However, most of the existing works for IPT detection only concern monophonic music signals, yet little has been done to detect IPTs in polyphonic instrumental solo pieces with overlapping IPTs or mixed IPTs. In this paper, we formulate it as a frame-level multi-label classification problem and apply it to Guzheng, a Chinese plucked string instrument. We create a new dataset, Guzheng\_Tech99, containing Guzheng recordings and onset, offset, pitch, IPT annotations of each note. Because different IPTs vary a lot in their lengths, we propose a new method to solve this problem using multi-scale network and self-attention. The multi-scale network extracts features from different scales, and the self-attention mechanism applied to the feature maps at the coarsest scale further enhances the long-range feature extraction. Our approach outperforms existing works by a large margin, indicating its effectiveness in IPT detection.

``` -->
[Paper on ArXiv]()
[Conference page]()
### Abstract of the paper

Instrument playing technique (IPT) is a key element of musical presentation. However, most of the existing works for IPT detection only concern monophonic music signals, yet little has been done to detect IPTs in polyphonic instrumental solo pieces with overlapping IPTs or mixed IPTs. In this paper, we formulate it as a frame-level multi-label classification problem and apply it to Guzheng, a Chinese plucked string instrument. We create a new dataset, Guzheng\_Tech99, containing Guzheng recordings and onset, offset, pitch, IPT annotations of each note. Because different IPTs vary a lot in their lengths, we propose a new method to solve this problem using multi-scale network and self-attention. The multi-scale network extracts features from different scales, and the self-attention mechanism applied to the feature maps at the coarsest scale further enhances the long-range feature extraction. Our approach outperforms existing works by a large margin, indicating its effectiveness in IPT detection.

#### Motivation
Guzheng is a polyphonic instrument. In Guzheng performance, notes with different IPTs are usually overlapped and mixed IPTs that can be decomposed into multiple independent IPTs are usually used. Most existing work on IPT detection typically uses datasets with monophonic instrumental solo pieces. This dataset fills a gap in the research field.

#### Description
We built a new dataset named **Guzheng_Tech99**  to conduct the analysis.

The dataset comprises 99 Guzheng solo compositions, recorded by professionals in a studio, covering various genres of Guzheng music and totaling 9064.6 seconds. It includes seven playing techniques labeled for each note (onset, offset, pitch, vibrato, point note, upward portamento, downward portamento, plucks, glissando, and tremolo), resulting in 63,352 annotated labels. The dataset is divided into 79, 10, and 10 songs for the training, validation, and test sets, respectively.

The paper will be released at ICASSP 2023 (IEEE International Conference on Acoustics, Speech, and Signal Processing).

More to be appeared……
