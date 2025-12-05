# Linear stimulus reconstruction for auditory attention decoding on the AV-GC-AAD dataset

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations. By downloading and/or installing this software and associated files on your computing system you agree to use the software under the terms and condition as specified in the License agreement.

If this code has been useful for you, please cite [1] (paper) and [2] (dataset).

## About

This repository includes the MATLAB-code to reproduce all experiments and results of the linear stimulus reconstruction approach for auditory attention decoding (AAD) on the publicly available KU Leuven audiovisual, gaze-controlled AAD (AV-GC-AAD) dataset [2], as presented in Geirnaert et al. [1].

The steps to reproduce the results:
1. Download the AV-GC-AAD dataset from [Zenodo](https://zenodo.org/records/11058711).
2. Fill in the correct data path to the dataset in the parameter settings of [mainSubjectSpecific.m](mainSubjectSpecific.m) (for subject-specific decoding) and [mainSubjectIndependent.m](mainSubjectIndependent.m) (for subject-independent decoding). 

Developed and tested in MATLAB R2021b.

Note: Tensorlab is required (https://www.tensorlab.net/).

## Contact
Simon Geirnaert  
KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics  
KU Leuven, Department of Neurosciences, Research Group ExpORL  
Leuven.AI - KU Leuven institute for AI  
<simon.geirnaert@esat.kuleuven.be>

Alexander Bertrand
KU Leuven, Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics  
Leuven.AI - KU Leuven institute for AI  
<alexander.bertrand@esat.kuleuven.be>

 ## References
 
[1] S. Geirnaert, I. Rotaru, T. Francart and A. Bertrand, "Linear stimulus reconstruction works on the KU Leuven audiovisual, gaze-controlled auditory attention decoding dataset," arXiv, 2024, [doi.org/10.48550/arXiv.2412.01401](https://arxiv.org/abs/2412.01401).

[2] I. Rotaru, S. Geirnaert, T. Francart and A. Bertrand, "Audiovisual, Gaze-controlled Auditory Attention Decoding Dataset KU Leuven (AV-GC-AAD)", Zenodo, 2024, [doi.org/10.5281/zenodo.11058711](https://zenodo.org/records/11058711). [dataset]
