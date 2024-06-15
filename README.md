# VarDial2024

<img width="200" alt="Screenshot 2024-06-11 at 14 15 58" src="https://github.com/StellaVerkijk/VarDial2024/assets/62950143/14da9c12-5775-49f3-aae5-f94b5ae2ae1b"> \


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


<img width="200" alt="Screenshot 2024-06-12 at 19 18 16" src="https://github.com/StellaVerkijk/VarDial2024/assets/62950143/b821ab19-2655-41c0-b805-ef0693f67cb1">


  



