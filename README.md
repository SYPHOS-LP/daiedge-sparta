# dAIEDGE - SPARTA

This repository holds code for the software and hardware activities of the dAIEDGE SPARTA project.
The repository is organized as 


The SPARTA project aims to develop and demonstrate a streamlined approach for deploying Vision 
Transformers (ViTs) on SoC FPGAs for real-time inference, based on both software and hardware 
adaptations. Central idea of SPARTA is the development of a ViT variant that employs end-to-end
sparsity both in model weights and activations, which are then utilized by an FPGA compute engine
that supports specialized sparse-dense matrix multiplication (SpDMM) and sparse-sparse matrix 
multiplication (SPMM) computation primitives.

Vision Transformers (ViTs) have been established as state-of-the-art models for a wide range of 
computer vision tasks. However, while ViTs can be efficiently deployed on accelerators such as 
GPUs and TPUs, their large size and high computational demands, make their deployment on embedded 
and edge devices considerably more challenging. Existing FPGA-based approaches primarily target 
data-center grade platforms, leveraging large on-chip memory capacities, ample compute resources, 
and specialized components such as High Bandwidth Memory (HBM), whereas analogous work targeting 
resource-constrained SoC FPGAs remains limited. 

SPARTA is a streamlined approach for deploying ViTs on small- and mid-range SoC FPGAs for real-time
inference through joint software and hardware adaptations. At the core of SPARTA is the development
of a ViT variant that employs end-to-end sparsity and quantization both in model weights and activations.
These are directly utilized by a dedicated FPGA compute engine that supports sparse-dense (SpDMM) and
sparse-sparse (SPMM) matrix multiplication  primitives. We deploy and evaluate SPARTA on the AMD Kria
KR260 board, achieving real-time inference throughput, with a modest drop in model's accuracy.


## Acknowledgements

🇪🇺 The SPARTA project has received financial support through the dAIEDGE project under its Financial 
Support to Third Parties (FSTP) scheme with **Grant Agreement No. dAI3OC03**. dAIEDGE is funded by 
the European Union's Horizon Europe research and innovation programme under Grant Agreement No. 101120726”

*Disclaimer: Funded by the European Union. Views and opinions expressed are however those of *
*the author(s) only and do not necessarily reflect those of the European Union or European Commission.*
*Neither the European Union nor the granting authority can be held responsible for them.*
