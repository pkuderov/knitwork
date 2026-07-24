#set page(
  width: 400pt,
  margin: (x: 8mm, y: 6mm),
)
#set text(
  // font: "Inter",
  size: 8pt
)

#let strength(body) = text(fill: rgb("#3b9a2c"), weight: "bold", body)
#let weakness(body) = text(fill: rgb("#b22222"), weight: "bold", body)

= A Modular RNN Architecture with Inter-Column Attention: Concept and Motivation

== Motivation

This sketch explores a modular recurrent architecture developed as a response to some limitations of standard transformer-based models, particularly in online, agent-centric scenarios such as reinforcement learning. Transformer architectures typically assume access to a fixed-length input sequence and offer powerful, global attention mechanisms. However, this comes at a cost: sequence-wide attention is expensive in both time and memory and not obviously suitable for incremental, real-time settings. The rigid structure also makes it hard to separate perception, planning, and control as interacting but distinct subsystems. Transformers work well when you can afford to batch everything and train offline, but they feel off when the agent needs to continually interact with a world.

This motivated a rethink of how to retain temporal processing capabilities (as in RNNs) while adding learnable information routing (as in attention-based models) in a modular way. Instead of unrolling a single monolithic RNN, the idea is to run several RNN columns in parallel, each with multiple layers, and allow them to communicate through attention.

== The Proposed Model

The following description complements the attached PDF sketch, which serves as a visual shorthand for explaining the core structural components of the model. While the sketch presents the architecture using diagrams and page-wise annotations, this text expands upon the motivations, implications, and intended usage. Throughout the text, I refer to specific points in the PDF sketch (e.g., [PDF, p.3]) to anchor visual references.

The architecture consists of N parallel RNN columns, each L layers deep *[PDF, p.1-3]*. Each column processes its own hidden state over time. Before each layer's RNN computation, attention mechanisms allow communication across columns within that layer *[PDF, p.4-5]*. This intra-layer, inter-column attention is the key mechanism for interaction, enabling information routing between specialsed modules without hardcoded pathways.

In the current proposal, I focus on intra-column, within-layer attention (horizontal attention). However, other forms of mixing can be explored in future variants, such as hierarchical attention within columns or irregular attention patterns that don't conform strictly to horizontal or vertical layout.

Columns can be designated as "bound" (receiving direct input or producing direct output) or "free" (no fixed external role, able to self-organize) *[PDF, p.6.1-6.3]*. This naturally supports multimodal input and output: different bound columns can receive different components of the observation, such as camera images, proprioceptive data, or textual instructions. Similarly, different output columns can control various modalities, including physical actuators, textual responses, or attention signals. The parallel column design allows simultaneous processing of multiple input streams, their fusion and selective entanglement or disentanglement, which is crucial for agents acting under rich and multimodal conditions *[PDF, p.7-7.1]*.

We can describe four connection types in the architecture:

+ forward temporal connections within each RNN cell, 
+ hierarchical vertical connections within a column (layer-to-layer), 
+ horizontal within-layer attention across columns, and
+ explicit connections from one column's output to another column's input. 

The fourth connection type, while loosely defined in the current version, opens up a direction toward feedback and control routing. Additional backward or cross-level connections (e.g., top-down modulation, vertical within-column attention, or mid-layer feedback) may be added later, though even a basic top-to-bottom feedback scheme is already expressive.

A particular form of attention is an open question that requires further exploration. The simplest approach is to consider all columns to be from the same modality, allowing them to attend to each other freely via self-attention. However, we may consider some sort of columns factorisation into distinct modalities and use cross-attention (e.g. each modality has its own query matrix at each layer, which is shared between cells of this modality within the layer). That's why I refer to this Throughout the text as simply attention, without specifying the exact form.

== Why This Design

This architecture aims to balance recurrence, modularity, and learnable communication. Unlike transformers, it doesn’t assume access to all inputs at once. Unlike vanilla RNNs, it avoids bottlenecking all computation through a single stream. Instead, it creates space for concurrent, possibly specialized, processing pathways. The model encourages the emergence of modular structure through training: different columns may learn to focus on different modalities, time scales, or tasks.

Some design choices are intentionally left flexible. Columns may have shared or separate parameters. Communication can be gated or open. More advanced routing patterns or feedback schemes can be introduced later. These degrees of freedom allow tailoring the model to different scenarios while preserving its core philosophy: recurrent modularity with learnable interconnectivity.

== Usage and Testing Strategy

To evaluate the model, one can begin with simple setups. A single column receiving an observation and producing an action is a natural starting point. Then, additional columns can be added without assigned roles to test whether the system learns to utilize them effectively. Tasks can be unimodal and non-interactive at first (e.g., sequence prediction or memory benchmarks), gradually moving to more complex agent-environment interactions. Useful baselines include: single RNN, transformer, and the same model with disabled attention.

In early versions, it makes sense to observe whether different columns specialize, how attention patterns evolve, and whether the model performance improves with added capacity. Adding auxiliary self-supervised losses (e.g. next-observation prediction) may encourage more effective use of free columns.

== Related Work and Conceptual Position

Several existing architectures touch on aspects relevant to this proposal:

- *Highway Networks* and *Multidimensional/Stacked LSTMs* (#link("https://arxiv.org/abs/1505.00387")[Srivastava et al. 2015], #link("https://arxiv.org/abs/1303.5778")[Graves et al. 2013]) propagate along both time and hierarchy but lack breadth (parallelism) and use fixed concatenation instead of learnable attention.
- *Grid LSTM* (#link("https://arxiv.org/abs/1507.01526")[Kalchbrenner et al. 2015]) introduces multi-axis flow using concatenated hidden states, not attention-based communication.
- *RWKV*, *RetNet*, and *SSM-based models* (like Mamba, Hyena) blend recurrence with attention but do so within a flattened sequence or channel-wise structure, rather than in a parallel modular RNN setup.
- *Self-Attending RNNs* (#link("https://pmc.ncbi.nlm.nih.gov/articles/PMC9560045/pdf/nihms-1797530.pdf")[Qin et al. 2022]) inject self-attention blocks into individual RNN layers to refine hidden states.
- *RNNs with Parallel Cells* (#link("https://arxiv.org/abs/1705.01346")[Kuchaiev et al. 2017]) use multiple smaller RNN cells instead of one large cell, but without attention-mediated interaction.
- #weakness()[*Relational Memory Core (RMC)*] (#link("https://arxiv.org/abs/1806.01822")[Santoro et al. 2018, DeepMind]) replaces a single hidden state with a set of memory slots interacting via attention. This shares some spirit with the proposed model, especially in multi-head recurrent attention across memory units.
- #weakness()[*Recurrent Independent Mechanisms (RIMs)*] (#link("https://arxiv.org/abs/1909.10893")[Goyal et al. 2019, Bengio and Levine]) provide a particularly close approach, with modules that operate independently and attend to each other sparsely. This model shares the idea of modular recurrent components connected via attention, though RIMs often enforce sparsity and activation gating (which I also have in mind among natural future directions).
- *Perceiver IO* also relates, in its flexible, modular I/O processing framework with efficient multi-modal interactions, though it does not focus on temporal recurrence or layered hierarchy.

Overall, while these works offer valuable insight and motivation, the proposed model blends recurrence, parallel modular structure, and intra-layer attention in a configuration that we have not seen explored directly.

== Potential Pitfalls

There are still open questions around self-specialization and gradient flow. Without proper incentives or architectural guidance, the model may ignore certain columns or fail to develop meaningful dependencies between them. Techniques such as regularization, auxiliary losses, or a gradual curriculum (e.g., starting with a single column and progressively adding more) might help address this. It’s also possible that excessive flexibility—such as allowing all columns to operate freely—could lead to instability or inefficiency, especially on simpler, single-goal tasks. From a purely engineering standpoint, many “how” questions remain unanswered.

In summary, this model is a sketch of a possible direction that combines the recurrence and streaming nature of RNNs with the compositionality and flexibility of attention, aiming to support scalable, agent-centric, and multimodal processing.
