
# `LSTM` based Pointer Network
This folder contains all code and all experiments connected to the neural network part of this reserach project
## Contents:
1. [Folders](#folders)
2. [Networks](#networks)
3. [Results](#results)
4. [Continued work](#continuation)


## Folders:
1. __[Experiments-folder](experiments)__: contains a folder for every empirical experiment that has been worth saving the result for. The experiment contains one folder for every single experiment labeld `"nbr_X"` where X is an integer. The experiment "nbr_X"- folder contains `plots`, `network architecture`, the `model-weights` and other `results` from that specific experiment.

2. __[Networks-folder](networks)__: Stand‑alone Python definitions of the different architectures we explored:

   The three different architectures that were used were:
    1. `pointer_lstm.py` – Baseline LSTM‑based Pointer Network
    2. `pointer_transformer.py` – Fully‑Transformer Pointer Network
    3. `pointer_hybrid.py` – Hybrid Self‑Attention + Pointer Network





## Networks


### 1 · LSTM Pointer Network (baseline)

Our original model is a classic **Pointer Network** with LSTM encoder and decoder. It processes an adjacency matrix, embeds each node, encodes the sequence with an LSTM, and then decodes node-by-node. At every decoding step the model points to either the next node or a special *EOS* symbol, thereby emitting one half of the Max-Cut partition.

__Highlights__
* `Input` – adjacency row per node → learned embedding.  
* **Encoder** – bidirectional LSTM captures global graph context.  
* **Decoder** – unidirectional LSTM; pointer (dot-product) attention over encoder states.  
* **Training** – teacher forcing with cross-entropy; explicit masking prevents the model from selecting a node twice.  
* **Inference** – greedy (or beam) search over the pointer distribution.  

---

### 2 · Transformer Pointer Network

This rewrite keeps the pointer idea intact but swaps both LSTMs for **multi-head self-attention** blocks:

* **Encoder** – 2-layer Transformer encoder encodes all nodes in parallel.  
* **Decoder** – Transformer decoder with causal self-attention plus encoder–decoder attention.  
* **Pointer** – at each step we dot the current decoder output against all encoder outputs (+ a learned *EOS* vector) to obtain pointer logits.

The interface is **identical** to the LSTM model (same input tensors, same pointer-logit output shape), so existing training scripts need only the import path changed.

---

### 3 · Hybrid Self-Attention Pointer Network

The hybrid variant keeps the **sequential pointer mechanism** of the LSTM model but replaces every LSTM cell with a **single-layer masked self-attention block** and a small feed-forward network:

* Each decoding step runs one masked multi-head attention over the embeddings of all previously selected nodes.  
* The resulting decoder state queries the (Transformer-encoded) node representations exactly like in the baseline.  
* A running mask still forbids the model from re-selecting the same node or EOS twice.  

Because only the internal cell changed, no training-loop or data-format changes are required.

---



## Results:
The best-performing network achieves just over `80%` accuracy in partitioning all nodes correctly on graphs with `100 nodes`. This means that, for these graphs, the model correctly classifies every node into its respective group more than 80% of the time. This network is saved in `"saved_models/n=100_82%.pth"`.

Other sizes of graphs we tried was `10` and `50` which we achieved an accuracy of over `97%` on.

We didnt try with a network bigger that a 100 nodes. And we only trained the networks on a CPU for at most couple hours.

---


## Continuation
I believe if we trained on a GPU for enough time that we could achieve near `100%` on the graph with 100 nodes, and possibly even larger.

An interesting thing to do could be to test it on the `big mac` data to compare it to the paper about maxcut in the `papers/maxcut` folder