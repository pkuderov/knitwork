= Knitwork directions

*Transportation*: how information and learning signals propagate, persist, and transform through time
+ dynamical sparse computational graph: shorter effective paths [through time and space]
  - dilated (time), residual (space) connections
  - update/reset (time) gating (as in LSTM/GRU), highway (space) connections (=between layers)
  - novelty-based gating
    - not only weighting, but binary gating
    - e.g. correct prediction vs error-informed paths

+ sparse activations (+weights?): less representation/gradient interference
  - SDR: sparsity via variational dropout + distributed (=redundand) via discrete dropout
  - or/and via explicit regularization

+ activations are important:
  - linear: easier grad flow
  - tahn nonlinearity is stable but has saturation regions, relu has reversed props

+ eprop + autograd to break TBTT horizon restriction:
  - to trade-off gradients precision with soft long horizons
+ initialization is important: identity, orthogonal/unitary
+ hiddens normalization: layer norm over batch norm
+ attention via actions: active seeking
+ problems to look after:
  - hidden state drift into saturation regions, unstable attractors, useless representations
  - stable gradients vs criticality:
    - stable grads lead to simple representations and help long-term memory preservation
    - criticality helps fast, sensitive adaptation, but is unstable

*Coordination*: how information is routed between components at a given time
+ dynamical sparse computational graph
  - response shortcuts: fast ans slow response systems (system 1 and 2)
  - can be learned by the energy minimization

+ top-down control:
  - explicit MP aka backward connections
  - activation, learning modulation

*Specialization*: how roles and responsibilities are distributed across components
+ information bottlenecks
  - smaller hiddens, less rich computations, partial information
  - time-delay for different parts, e.g. layer-to-layer one-step delay (model message propagation times)
  - random blind spots or entirely blind steps to stress memory formation and usage

+ frozen/hebbian parts w/ backptop-learned communication/readout between them
  - ESN/reservoir computing: rich dynamics, hard to tune initialization, passive memory, no gating or control
+ hierarchical abstraction formation
  - aux summary / compression losses?

+ explicit energy consumption (=computational work) notion
  - reduce duplication of roles
  - minimize the computations/activity, i.e. dynamic computation path and/or force sparse weights/activations
  - but do not forget about noise to keep some duplication

+ evolutionary selection of connectivity (short genes)

*Richness*: how expressive the representations and computations are
+ temporal basis / memory parameterization mechanisms
  - fast weights -like attention
    - FW is a specific past aggregation case (cross-element corellation), we can consider other options too
  - interconnect past and future EMA aggregations:
    - it is usual to track EMA of the past states
    - we can also learn symmetrically EMA of the future — SFs
  - random, learned, modulated gammas or learning rates
    - fast & slow learning for LRs
    - mixture of effective horizons for LRs, EMA gammas
    - connects with EMA/SF learning: per-neuron random gamma (=1-lr)
    - connects with Fast Weigts: again, can be per-neuron random horizon accumulation
  - hippo-like basis
+ uncertainty / belief representation
  - explicit Dreamer-like deterministic + sampled stochastic parts of the state
  - SDRs has a clean, compact support of superpositions
+ intra-timestep recurrency:
  - can be local to cell, or more global (entire network) akin to the chain-of-thoughts
  - efficiency: some of the "deep layered cell chains" can be replaced by the single recurrent cell unroll with sparse asymetric forward dynamics
  - iterative reasoning
+ equilibrium RNNs: energy/error-based local-time iterative convergency, attractors
