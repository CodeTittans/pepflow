# PepFlow

![Pepflow schematic](/imgs/banner.png "Modularized peptide generation with PepFlow")

## Description

PepFlow is a deep-learning method for direct all-atom peptide conformational sampling. The model is a Boltzmann generator trained in a diffusion framework and subsequently used as a flow for training by energy and sampling. PepFlow contains a number of provisions that enable the modelling of all the degrees of freedom in a peptide conformation. The details of this approach can be found in the associated preprint:

Abdin O, Kim PM. Pepflow: direct conformational sampling from peptide energy landscapes through hypernetwork-conditioned diffusion. bioRxiv. 2023.

## Installation

To install PepFlow and the necessary dependancies run the following commands:

1. Clone the repository `git clone https://gitlab.com/oabdin/pepflow.git`
2. Create the conda environment `conda env create -f pepflow_env.yml`
3. Activate the conda environment `conda activate pepflow_env`
4. Install remaining requirements:\
cuda-compiled version of torch-scatter `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html`\
ml-collections `pip install ml-collections==0.1.1`
5. Install the pepflow package `pip install .`

To run PepFlow, download and extract the model parameters, http://pepflow.ccbr.proteinsolver.org/params.tar.gz.

## Peptide structure prediction and ensemble generation

To predict peptide structures use the `generate_peptide_samples.py` with the following options:

`-fm` full model parameters file, either this parameter needs to be specified or all of `-bm`, `-rm` and `-hm` need to be specified\
`-bm` backbone model parameters file\
`-rm` rotamer model parameters file\
`-hm` hydrogen model parameters file\
`-o` output directory

`-n` number of samples to generate\
`-c` number of samples per chunk to use when generating the samples. This value can be reduced to decrease memory usage at the cost of slower sampling, recommended value is 10

`-s` the sequence for which samples should be generated

`--s` flag to generate a single structure prediction

`--r` flag to not generate side chain heavy atoms\
`--p` flag to not protonate generated peptides\
`--f` flag to not correct the chirality of generated peptides\
`--d` flag to generate D-amino acid peptides (if `--f` is not set)

`--l` flag to output likelihoods of generated peptides\
`--e` flag to output energies of generated peptides

### Examples

The following command generates 100 conformations for the peptide LQTKLKKLLGLESVF using PepFlow after training by energy, takes 3-4 minutes on an NVIDIA GeForce RTX 2070 SUPER:

`python generate_peptide_samples.py -s LQTKLKKLLGLESVF -o LQTKLKKLLGLESVF_samples --e -fm params/full_model.pth -n 100 -c 10`

The following command generates 100 conformations and a single-structure prediction for the peptide LKKLWRFLKKL using PepFlow that finetuned on peptide structures, takes 2-3 minutes on an NVIDIA GeForce RTX 2070 SUPER:

`python generate_peptide_samples.py -s LKKLWRFLKKL -o LKKLWRFLKKL_samples --s -bm params/params_backbone_finetuned.pth -hm params/params_hydrogen.pth -rm params/params_rotamer.pth -n 100 -c 10`

Example outputs are provided in the `examples/` folder.

## Macrocyclic peptide conformation prediction

To predict macrocyclic peptide conformations use the `generate_cyclic_peptide_samples.py` script with the following options:

`-fm` full model parameters file, either this parameter needs to be specified or all of `-bm`, `-rm` and `-hm` need to be specified\
`-bm` backbone model parameters file\
`-rm` rotamer model parameters file\
`-hm` hydrogen model parameters file\
`-o` output directory

`-n` number of samples to generate\
`-c` number of samples per chunk to use when generating the samples. This value can be reduced to decrease memory usage at the cost of slower sampling, recommended value is 5\
`-st` number of MCMC steps, recommended value is 250

`-s` the sequence for which samples should be generated\
`-b` a file with a list of cyclic bonds. Each line in the file should contain two atoms separated by a comma, with each atom expressed as the amino acid number and the atom name separated by an underscore. Example files can be found in the `examples/` folder

`--s` flag to generate a single structure prediction

`--f` flag to not correct the chirality of generated peptides\
`--d` flag to generate D-amino acid peptides (if `--f` is not set)

### Examples

The following command generates 25 conformations and a single structure prediction of the peptide GRCTKSIPPRCFPD with head-to-tail cyclization, takes ~13 minutes on an NVIDIA GeForce RTX 2070 SUPER:

`python generate_cyclic_peptide_samples.py -s GRCTKSIPPRCFPD -b examples/bonds_GRCTKSIPPRCFPD.txt -c 5 -n 25 --s -o GRCTKSIPPRCFPD_samples  -fm params/full_model.pth -st 100`

The following command generates 25 conformations and a single structure prediction of the peptide EADKWQS with cyclization between the C-beta of the A2 and NE2 of Q6, takes ~22 minutes on an NVIDIA GeForce RTX 2070 SUPER:

`python generate_cyclic_peptide_samples.py -s EADKWQS -b examples/bonds_EADKWQS.txt -c 5 -n 25 --s -o EADKWQS_samples  -fm params/full_model.pth` 

Example outputs are provided in the `examples/` folder.

## Datasets

Datasets and processed features for training an evaluation can be downloaded from http://pepflow.ccbr.proteinsolver.org/. If `md_pdbs` is used please cite DBAASP 3.0 (https://doi.org/10.1093/nar/gkaa991) and if PED-derived peptides are used please cite the PED (https://doi.org/10.1093/nar/gkaa1021). Scripts for training and evaluation are present in the `scripts` folder.
