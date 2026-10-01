# Data sources

This directory does not contain copies of external structural or sequence
data. Raw structures and reference databases are fetched directly from
their sources below; only derived, study-specific files (target lists,
leakage-audit outputs, computed coordinates) are stored under `cohort/`.

## Structures

RCSB Protein Data Bank: https://www.rcsb.org
Individual entries: https://www.rcsb.org/structure/<PDB_ID>

## Training-corpus / leakage-audit reference

RNA3DB, release 2024-04-26: https://github.com/marcellszi/rna3db

## Primary backbone

RNAbpFlow: https://github.com/Bhattacharya-Lab/RNAbpFlow
Code archive: https://doi.org/10.5281/zenodo.19388910

## H2 candidate backbones

Boltz-2: https://github.com/jwohlwend/boltz
Pair representation scaling: https://doi.org/10.1021/acs.jcim.6c02094

## Secondary-structure conditioning tools

MXfold2: https://github.com/mxfold/mxfold2
CONTRAfold: via Bioconda
SimFold: built from MultiRNAFold-2.1 source
IPknot: https://github.com/satoken/ipknot
