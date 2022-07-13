# A GPT-like Model

I build a GPT-like model from scratch, and the model is adapted for pipeline training.

This repo can be used as an example to learn the pipeline mechanism of colossal-ai.

## Requirement
1. A slurm environment  
   I run this model on the computing cluster of my university,
   but it can be easily adapted to other servers by modifying the shell script.
    
2. [Colossal-AI](https://github.com/hpcaitech/ColossalAI)  
   To split an extremely large model by layers and put each part into different GPU (i.e., pipeline parallelism),
   I adopt the utility modules provided by Colossal-AI, and manually split the model.
    
3. [Transformers](https://github.com/huggingface/transformers)  
   It's quite convenient to build a Byte-level Byte Pair Encoding (BBPE) tokenizer with Transformers' pre-trained tokenizer models.
   
   I also write a python script which uses Transformers' `tokenizers` module to train a BBPE tokenzier model over corpora; 
   refer to the `tokenizer` directory to see details.
   
## Dataset
Colossal-AI provides a [GPT example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt) accompanied with scripts to download and preporcess OpenWebText.  

However, thanks to Transformers again, you can easily download a clean OpenWebText by Transformers' Datasets;
for the full OpenWebText, refer to [https://huggingface.co/datasets/openwebtext](https://huggingface.co/datasets/openwebtext).

Here, I adopt the [first 10k passages](https://huggingface.co/datasets/stas/openwebtext-10k), and further extract the first 16 passages into `toy.json` to check the sanity of the whole running process.

## Model
I refer to [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) to build the transformer decoder layers from scratch.  
I adapt the model for pipeline training with the help of the Colossal-AI's [model zoo](https://github.com/hpcaitech/ColossalAI/tree/main/model_zoo).

There are several differences need to be noted:
1. I adopt randomly initialized trainable positional embedding according to OpenAI GPT.
2. The self-attention layer and fully connected layer are similar to those of the original transformer decoder.

## Parallelism Config & Command
As required by Colossal-AI, the configuration for parallel training and hyperparameters is defined in `config.py`.  

For the running commands, please refer to `run.sbatch`.  

To obtain the host ip for multiple nodes in a slurm-based cluster, I refer to [
distribuuuu](https://github.com/BIGBALLON/distribuuuu).
