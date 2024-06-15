# VarDial2024

<p> This repository contains code and data to run experiments as well as a collection of annotated data presented in the paper presented at VarDial, NAACL 2024 (June), by Verkijk, Sommerauer and Vossen (_Studying Language Variation Considering the Re-Usability of Modern Theories, Tools and Resources for Annotating Explicit and Implicit Events in Cnturies Old Text_).This work is part of the GLOBALISE project. </p>

### annotated_data

This folder contains all annotated data collected thus far for event detection and classifcation within GLOBALISE. 

- **train**
  - **train_2** Documents annotated by trained annotators in Round 2 as described in the paper -  _54 pages_
  - **train_3** Documents annotated in Round 3 as described in the paper -  _57 pages_
- **test**
  - **curated** One document annoated and subsequently curated by four historians and a linguist - _5 pages_
  - **non-curated** Two documents, one annotated in Round 2 and one in Round 3, annotated by two and four annotator teams respectively, that are to be curated to serve as an addition to the test set. _21 pages_

The documents included in **non-curated** are also those used for calculating the IAA. 
The documents included in **train_2** are also used in the LLM-finetuning experiment, where this data is split in train and test. 


  



