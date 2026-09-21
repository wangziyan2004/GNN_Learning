# GNN Learning

> 📚 A personal learning repository for **Graph Neural Networks (GNNs)**
> 🧑‍💻 Python + PyTorch + PyTorch Geometric
> 📝 Learning Notes: OneNote + Obsidian
> 🤖 AI Assistant: DeepSeek-V4-Pro
> 🖥️ Remote Environment: Ubuntu Server + MobaXterm

---

## 📖 About This Repository

This repository records my learning process in **Graph Neural Networks (GNNs)**, including:

* 📐 Mathematical foundations
* 🕸️ Graph theory fundamentals
* 🧠 GNN basic concepts
* 🔄 Message Passing
* 🧩 Classic GNN models
* 💻 PyTorch / PyG implementations
* 🧪 GNN experiments
* 📝 Learning notes and summaries

The main goal is to combine:

> **Theory → Manual Calculation → Code Implementation → Experiments → Notes**

Instead of only reading the theory, I try to understand each model through mathematical derivation, small-scale manual calculations, and actual implementation.

---

## 🛠️ Technology Stack

### Programming

* Python
* PyTorch
* PyTorch Geometric (PyG)

### Development Environment

* Ubuntu Server
* MobaXterm
* Anaconda
* Git
* GitHub

### Learning & Note-Taking

* OneNote
* Obsidian

### AI Assistant

* DeepSeek-V4-Pro

---

## 🧠 Learning Roadmap

The current learning roadmap is roughly organized as follows:

```text
Mathematical Foundations
        │
        ▼
    Graph Theory
        │
        ▼
   GNN Fundamentals
        │
        ├── Node Classification
        ├── Graph Classification
        └── Graph Pooling
        │
        ▼
 Message Passing Framework
        │
        ▼
 ┌──────────────────────┐
 │      Classic GNNs    │
 ├──────────────────────┤
 │ GCN                  │
 │ GraphSAGE            │
 │ GAT                  │
 └──────────────────────┘
        │
        ▼
   PyTorch Geometric
        │
        ▼
   GNN Experiments
        │
        ▼
Advanced GNN Topics
```

---

## 📚 Learning Content

### 1. Graph Theory

Fundamental graph concepts required for understanding GNNs:

* Graph
* Node / Vertex
* Edge
* Degree
* Adjacency Matrix
* Degree Matrix
* Path
* Neighborhood
* Directed / Undirected Graph
* Weighted Graph
* Self-loop

---

### 2. GNN Fundamentals

Core concepts of Graph Neural Networks:

* Node Features
* Graph Structure
* Message Passing
* Message
* Aggregate
* Update
* Neighborhood Aggregation
* Layer-wise Propagation

The basic idea can be summarized as:

```text
Neighbor Information
        │
        ▼
     Message
        │
        ▼
    Aggregate
        │
        ▼
      Update
        │
        ▼
New Node Representation
```

---

### 3. GNN Tasks

Main graph learning tasks:

#### Node Classification

Predict the label of each node.

```text
Graph
  │
  ▼
GNN
  │
  ▼
Node Embeddings
  │
  ▼
Classifier
  │
  ▼
Node Labels
```

#### Graph Classification

Predict a label for an entire graph.

```text
Nodes
  │
  ▼
GNN
  │
  ▼
Node Embeddings
  │
  ▼
Graph Pooling
  │
  ▼
Graph Representation
  │
  ▼
Classifier
  │
  ▼
Graph Label
```

#### Graph Pooling

Aggregate node-level representations into a graph-level representation.

---

## 🔬 Classic GNN Models

### GCN

Graph Convolutional Network.

Key ideas:

* Graph convolution
* Neighborhood aggregation
* Adjacency matrix
* Degree matrix
* Normalization
* Self-loop

Typical propagation:

$$
H^{(l+1)}
=
\sigma
\left(
\hat{D}^{-\frac12}
\hat{A}
\hat{D}^{-\frac12}
H^{(l)}
W^{(l)}
\right)
$$

---

### GraphSAGE

Graph Sample and Aggregate.

Key ideas:

* Neighborhood sampling
* Neighbor aggregation
* Inductive learning
* Mean / Pool / LSTM aggregation

Basic idea:

```text
Node
 │
 ├── Sample Neighbors
 │
 ▼
Aggregate Neighbor Features
 │
 ▼
Combine With Node Features
 │
 ▼
New Node Representation
```

---

### GAT

Graph Attention Network.

Key ideas:

* Attention mechanism
* Attention coefficients
* Neighbor importance
* Multi-head attention
* Self-loop

Basic process:

```text
Node Features
      │
      ▼
Calculate Attention
      │
      ▼
Attention Coefficients
      │
      ▼
Weighted Neighbor Aggregation
      │
      ▼
New Node Representation
```

---

## 🧮 Manual Calculation

An important part of my learning process is **manual calculation**.

For example, for a small graph:

```text
    A
   / \
  B---C
       \
        D
```

I will manually calculate:

1. Node features
2. Adjacency matrix
3. Degree matrix
4. Normalized adjacency matrix
5. Message passing
6. Aggregation
7. Node representation update
8. Activation
9. Classification output

The purpose is to understand **what the code is actually calculating** rather than treating GNN layers as black boxes.

---

## 💻 Code Organization

The repository will gradually organize implementations according to the learning process.

A possible structure:

```text
GNN_Learning/
│
├── README.md
│
├── graph_theory/
│   ├── graph_basic.py
│   └── adjacency_matrix.py
│
├── gnn_basic/
│   ├── message_passing.py
│   ├── node_classification.py
│   ├── graph_classification.py
│   └── graph_pooling.py
│
├── gcn/
│   ├── gcn_basic.py
│   └── cora_gcn.py
│
├── graphsage/
│   ├── graphsage_basic.py
│   └── cora_graphsage.py
│
├── gat/
│   ├── gat_basic.py
│   └── cora_gat.py
│
├── experiments/
│   ├── cora/
│   └── ...
│
└── notes/
    └── ...
```

The directory structure may change as the learning progresses.

---

## 🧪 Experiments

### Cora Node Classification

The first GNN experiment uses the **Cora** citation network dataset.

Basic workflow:

```text
Cora Dataset
     │
     ▼
Load Graph
     │
     ▼
Node Features
     │
     ▼
Graph Structure
     │
     ▼
GNN Model
     │
     ▼
Node Embeddings
     │
     ▼
Classification
     │
     ▼
Accuracy / Loss
```

The Cora experiment is used as an entry point for understanding how GNNs are implemented with **PyTorch Geometric**.

---

## 📝 Learning Notes

Theoretical knowledge and code are recorded separately.

### OneNote

Mainly used for:

* Detailed handwritten-style notes
* Mathematical derivations
* Manual calculations
* Learning process
* Important concepts

### Obsidian

Mainly used for:

* Structured knowledge management
* Concept relationships
* Markdown notes
* Knowledge graph
* Long-term organization

The general relationship is:

```text
Learning
   │
   ├── OneNote
   │     └── Detailed Study Notes
   │
   └── Obsidian
         └── Structured Knowledge Base
```

---

## 🤖 AI-Assisted Learning

During the learning process, **DeepSeek-V4-Pro** is used as an AI learning assistant.

Typical use cases include:

* Explaining difficult concepts
* Breaking down mathematical formulas
* Checking manual calculations
* Generating small examples
* Explaining PyTorch / PyG code
* Debugging experiments
* Comparing different GNN architectures
* Helping organize learning notes

The AI assistant is used as a **learning aid**, while the final understanding and verification are based on mathematical derivations, documentation, code execution, and experiments.

---

## 🖥️ Development Environment

The experiments are mainly conducted on an Ubuntu server.

```text
Windows PC
    │
    │ MobaXterm
    ▼
Ubuntu Server
    │
    ├── Anaconda
    │
    ├── Python
    │
    ├── PyTorch
    │
    ├── PyTorch Geometric
    │
    └── GNN Experiments
```

MobaXterm is mainly used for:

* SSH connection
* Remote terminal
* File management
* Running experiments
* Managing the Ubuntu environment

---

## 🔄 Learning Workflow

My current learning workflow is:

```text
① Learn Theory
      ↓
② Understand Formula
      ↓
③ Manual Calculation
      ↓
④ Implement With Python
      ↓
⑤ Run Experiment
      ↓
⑥ Analyze Results
      ↓
⑦ Organize Notes
      ↓
⑧ Commit to GitHub
```

This repository therefore serves as both a **learning record** and an **experimental codebase**.

---

## 🎯 Current Progress

### Completed / Learning

* [x] Python basics
* [x] Basic graph theory
* [x] GNN basic concepts
* [x] Node classification
* [x] Graph classification
* [x] Graph pooling
* [x] Message Passing
* [x] GCN
* [x] GraphSAGE
* [x] GAT
* [x] Basic PyTorch Geometric usage
* [x] Cora node classification experiment

### Next Steps

* [ ] More PyG experiments
* [ ] Compare GCN / GraphSAGE / GAT
* [ ] Understand GNN over-smoothing
* [ ] Understand heterophily
* [ ] Learn graph-level models in more depth
* [ ] Explore more advanced GNN architectures
* [ ] Read and reproduce representative GNN papers
* [ ] Build independent GNN experiments

---

## 📌 Learning Philosophy

The main principle of this repository is:

> **Don't just memorize the model. Understand what happens to the graph and node representations at every step.**

For each GNN model, I try to answer:

```text
1. What problem does it solve?
2. What is the mathematical formulation?
3. How does Message Passing work?
4. How are neighbors aggregated?
5. What is different from previous models?
6. What does the code actually calculate?
7. Can I manually calculate a small example?
8. Can I implement and run it myself?
```

---

## 🚀 Goal

The long-term goal is to build a systematic understanding of **Graph Neural Networks** and gradually transition from:

```text
Learning GNN
     ↓
Implementing GNN
     ↓
Experimenting With GNN
     ↓
Reading GNN Papers
     ↓
Reproducing GNN Research
     ↓
Conducting Independent Research
```

---

## 📚 References

* [Dive into Deep Learning](https://zh.d2l.ai/)
* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
* [Cora Dataset](https://graphsandnetworks.com/the-cora-dataset/)
* Original GNN / GCN / GraphSAGE / GAT papers

---

## ⭐ Repository Status

🚧 **Learning in Progress**

This repository will be continuously updated as my understanding of GNNs, PyTorch Geometric, and graph learning research develops.

> **Learn → Calculate → Implement → Experiment → Record → Research**
