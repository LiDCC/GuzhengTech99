# Frame-Level Multi-Label Playing Technique Detection Using <br> Multi-Scale Network and Self-Attention Mechanism
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
Guzheng is a polyphonic instrument. In Guzheng performance, notes with different IPTs are usually overlapped and mixed IPTs that can be decomposed into multiple independent IPTs are usually used. Most existing work on IPT detection typically uses datasets with monophonic instrumental solo pieces, which lacks the complexity present in music with overlapping IPTs or mixed IPTs. This dataset fills a gap in the research field and increases the scope and diversity of research in the field of instrument playing technique detection.

#### Description
We built a new dataset named **Guzheng_Tech99**  to conduct the analysis.

The proposed dataset consists of 99 audio recordings of Guzheng solo compositions recorded by two professional Guzheng players in a professional recording studio. The audio excerpts in the dataset cover most of the genres of Guzheng music. The audio recordings in the dataset are 9064.6 seconds long in total.

We consider seven playing techniques in our dataset (vibrato, point note, upward portamento, downward portamento, plucks, glissando, tremolo). We label the onset, offset, pitch, and playing techniques of every note in each recording.  As a result, the dataset consists of 63,352 annotated labels in total.    

The dataset is split into 79, 10, and 10 songs respectively for the training set, the validation set, and the test set.  

The paper will be released at ICASSP 2023 (IEEE International Conference on Acoustics, Speech, and Signal Processing).

