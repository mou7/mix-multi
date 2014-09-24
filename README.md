mix-multi
=========

This code implements the EM algorithm for the mixture of Multinomials. 
The generative process that explains the mixture of Multinomials: 
For each document, we sample a topic from a document-specific multinomial. 
Given this topic, we sample a word from the Multinomial distribution 
that represents this topic. 

running the code 
=================

To run the code with a synthetic dataset
``` python
python -i mix_multi.py
```
