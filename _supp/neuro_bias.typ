#set page(
  paper: "a4",
  margin: (x: 26mm, y: 25mm),
)
#set text(
  font: "Libertinus Serif",
  size: 11pt,
  lang: "en",
)
#set par(justify: true, leading: 0.65em)
#show heading.where(level: 1): set text(size: 16pt, weight: "bold")
#show heading.where(level: 2): set text(size: 12.5pt, weight: "bold")

#align(center)[
  #text(size: 20pt, weight: "bold")[
    Weak Neuroanatomical Biases for Modular Recurrent Systems
  ]

  #v(0.5em)
  A conceptual map and shared context for open-ended architectural thinking
]

#v(1em)

= 1. Status and intended use

This document records an early-stage research idea and a space of possibilities around it. It is neither a finished architecture nor a proposal for one, and it deliberately does not prescribe an implementation sequence, experimental program, or preferred minimal model. Its purpose is to support further thought before those choices become appropriate.

The central intuition is that neuroanatomy may offer a useful catalogue of weak computational inductive biases for modular recurrent systems. The useful object is not the biological structure itself, but the family of computational tendencies it brings to attention: selective connectivity, heterogeneous state geometry, different timescales, distinct update dynamics, local or global learning signals, associative binding, routing, prediction, selection, modulation, and structured recurrent loops.

This document has two related uses:

- to preserve potentially useful distinctions, interpretations, interactions, and cautions while the idea is still being explored;
- to provide shared context for collaborators and AI agents, so that discussion can begin from the intended level of abstraction without repeatedly correcting literal or prematurely concrete interpretations.

The material below should be read according to the following rules.

- *The biological names are handles, not contracts.* A thalamus-like component need not reproduce the thalamus, and a hippocampus-like process need not correspond to human episodic memory. The names point toward clusters of potentially useful biases.
- *Mention is not commitment.* A mechanism, area, learning signal, or interaction described here is part of the design space unless explicitly marked as selected. No such selection is made in this document.
- *Functions need not belong to single areas.* A function may emerge from a loop spanning several areas; one area may participate in several functions; several area classes may contribute similar computations.
- *Architectural and biological boundaries need not coincide.* An artificial implementation may split one biological motif across several components, merge several motifs, or realize the relevant computation in a very different substrate.
- *Mechanisms should remain multiply realizable.* Persistent state, fast weights, ordinary parameters, external memory, sparse activity, attention, multiplicative gating, temporary plasticity, and other substrates are alternatives to consider, not defaults inferred from a label.
- *Do not force premature convergence.* When using this document as context, the immediate task is normally to elaborate, compare, question, connect, or reinterpret possibilities. Do not turn every ambiguity into a decision, every motif into a module, or every idea into a narrow implementation or experiment unless explicitly asked.
- *Preserve productive tensions.* If several interpretations appear plausible, make their differences visible rather than selecting the easiest one to implement. If a concept is underspecified, clarify its degrees of freedom before optimizing it.

The desired mode of reasoning is therefore exploratory but disciplined. Biological resemblance is not evidence of computational value, yet implementation convenience is also not sufficient reason to collapse a broad idea into the nearest familiar machine-learning component. The aim at this stage is to understand the space well enough that later architectural commitments will be informed rather than accidental.

= 2. Core idea

Begin with a mostly general recurrent computational substrate capable of learning representations and transformations. Instead of making the entire substrate homogeneous, allow different parts of it to acquire or receive weak biases along several independent dimensions. Some parts may remain close to generic recurrent processors. Others may be predisposed toward sparse conjunctive state, persistent routing configurations, fast predictive correction, selective gating, slow integration, rapid adaptation, or low-dimensional modulation.

These predispositions should influence which solutions are easy, stable, or natural without fixing the semantic function a component must learn. In this sense, a *weak bias* is a pressure on the organization or dynamics of computation, not a manual assignment of meaning. It can be soft or hard in mechanism while remaining weak in semantic prescription: restricted connectivity, for example, may be structurally firm, yet still leave the connected components free to learn unexpected functions.

The model is best imagined not as a collection of named brain-region replicas but as a structured population of recurrent and non-recurrent processes participating in overlapping loops. Different loops may bind information, maintain context, regulate communication, predict transitions, select internal operations, or shape outputs. They may share some components and signals while preserving enough separation to reduce interference and support differentiated dynamics.

The neuroanatomical analogy matters because it suggests combinations of biases that do not arise naturally when thinking only in terms of uniform layers or globally mixed hidden states. It draws attention to questions such as:

- Should every state be equally writable, persistent, and visible?
- Should communication be a momentary content mixture or a persistent, structured regime?
- Should event-specific conjunctions and slowly accumulated statistical structure occupy the same representational substrate?
- Should every internal transformation occur automatically, or should some require context-sensitive selection?
- Should all parts of a model be taught by the same objective and on the same timescale?
- Can low-dimensional signals coordinate plasticity, gain, persistence, or allocation without carrying detailed content?
- Are recurrent cross-area loops a more useful unit of organization than a flat graph of modules?

These questions are more fundamental than any particular biological mapping. The mappings are useful insofar as they help maintain and refine the questions.

= 3. Architectural vocabulary

Several levels of description are needed to avoid turning a high-level diagram into a graph of monolithic nodes.

- An *area* is a broad family of components sharing a weak architectural or functional tendency. It may be internally heterogeneous, spatially distributed in the artificial graph, and involved in several loops.
- A *module* is a more cohesive local subsystem within an area. Modules may differ in inputs, outputs, state structure, update rate, connectivity, or learning signals even when they share the same broad area label.
- A *block* is a smaller computational unit considered explicitly. It may be recurrent, feed-forward, attention-based, memory-based, state-space-like, or something else.
- A *loop* or *channel* is a recurrent functional pathway crossing several areas, modules, or blocks. It may maintain relatively private state while exchanging selected information with other loops.
- A *signal* is an interaction that need not correspond to ordinary content transmission. Routing coefficients, prediction errors, value estimates, novelty, confidence, gain, plasticity, and write permission are different kinds of signals even when represented by similar tensors.

An area should therefore not be assumed to be one module, and an apparent area-to-area edge should not imply full broadcast. A connection may contain several factorized channels, carry different signal types in different directions, or belong to multiple partially independent loops. Conversely, an apparently specialized computation may arise from the interaction of general components rather than from one specialized node.

This vocabulary is provisional. Its purpose is to keep different scales of organization distinct, not to impose a permanent ontology.

= 4. Independent dimensions of bias

Neuroanatomical motifs often appear as bundles, but the elements of those bundles should remain conceptually separable. A useful way to inspect any proposed area or loop is to ask which of the following dimensions it actually changes.

== 4.1. Connectivity and communication

Connectivity bias concerns which components can exchange information, in which directions, through which channels, and under whose control. Communication may be dense or sparse, fixed or learned, direct or routed, symmetric or asymmetric, continuously available or conditionally enabled. A bias may also limit bandwidth, enforce locality, preserve private state, or allow a communication pattern to persist across several recurrent steps.

The important distinction is not simply connected versus disconnected. A system can vary the structure of possible communication, the momentary effective graph, and the content passed over each active edge. These are separate design degrees of freedom.

== 4.2. State and representation

State bias concerns the geometry and functional behavior of stored information. State may be dense or sparse, distributed or slot-like, smooth or conjunctive, rapidly changing or persistent, easily overwritten or selectively protected. Some representations may favor generalization across related situations; others may favor separation of similar events. Some may carry detailed content, while others summarize context, uncertainty, value, or control state.

No single geometry should be assumed to own a cognitive function. Sparse state is not automatically episodic memory, and dense state is not automatically semantic knowledge. The question is which representational pressures make particular computations easier and how different state types interact.

== 4.3. Update dynamics and timescales

Update bias concerns how a component changes within an episode or stream. Possibilities include ordinary recurrent replacement, gated preservation, additive accumulation, attractor-like settling, competitive allocation, predictive correction, associative writing, or modulation by another loop. Different variables may update at different rates, and their persistence may range from a fraction of a step to an entire task or beyond.

Timescale applies separately to activations, routing configurations, temporary memory, plastic variables, and ordinary parameters. A component can have rapidly changing activity but slowly changing routing, or persistent activity with rapidly adapting readout. Treating all of these as one notion of “memory length” loses useful structure.

== 4.4. Learning and plasticity

Learning-signal bias concerns what information shapes a component and when. A subsystem may be influenced by a global task objective, local prediction error, reconstruction, temporal-difference error, novelty, replay, consistency, uncertainty reduction, or signals generated elsewhere in the architecture. These influences may change ordinary parameters, temporary synaptic variables, recurrent state, routing policy, or plasticity itself.

It should remain open whether an apparent fast-learning function requires fast parameter change. Rapid storage may instead use persistent activations, writable state, external slots, fast weights, or another temporary substrate. Likewise, a local objective need not define the eventual semantic role of the component it trains.

== 4.5. Control and conditional computation

Control bias concerns whether a transformation occurs automatically, is continuously modulated, or requires explicit selection. The controlled object may be external behavior, but it may also be an internal operation: preserve a state, permit a write, retrieve a binding, open a route, allocate additional computation, change gain, or switch dynamical regime.

Control can be local or global, soft or discrete, myopic or value-sensitive. A useful distinction is between representing content and deciding what may happen to that content, although the two need not reside in completely separate components.

== 4.6. Learning to learn and operating regime

Some biases act on the way other parts of the system operate rather than on their immediate content. Low-dimensional signals may alter gain, update rate, exploration, noise, plasticity, persistence, or routing priorities. Such signals create the possibility of context-dependent operating regimes: stable retention versus rapid adaptation, routine processing versus broader recruitment, or exploitation versus exploratory search.

This dimension is especially easy to overcentralize. A global modulator may become a shortcut, bottleneck, or source of instability. Multiple limited modulatory channels may better preserve differentiated control.

= 5. Neuroanatomical motifs as sources of bias

The following motifs are not a final taxonomy. Each is a prompt for considering a cluster of computational properties, alternative interpretations, and interactions. Their boundaries overlap intentionally.

== 5.1. Cortex-like general recurrent processing

A cortex-like substrate denotes a broad population of relatively general recurrent processors that learn distributed representations and transformations. The label does not require cortical columns, laminar anatomy, specific cell types, or a fixed map of human cortical functions. Its main role in this conceptual picture is to supply flexible content-bearing computation that can be shaped by position in the graph and participation in different loops.

Even the general substrate need not be homogeneous. Its modules may differ weakly through their inputs, recurrent timescales, state dimensionality, sparsity, accessible learning signals, and communication partners. Sensory-oriented modules may be exposed to local modality-specific structure. Association-like modules may integrate information across streams, contexts, and timescales. Executive-like modules may maintain goals, rules, hypotheses, or intermediate states. Output-oriented modules may transform internal decisions into structured external signals.

These descriptions should be treated as statistical and relational pressures, not symbolic job assignments. A module's position and training conditions may encourage a role without determining it. Several executive-like processes may coexist, and association or control functions may be distributed across loops rather than localized.

An unresolved question is how much differentiation belongs inside the general substrate versus in interactions with more strongly biased components. A nearly uniform cortex-like population maximizes flexibility; a structured population may reduce the burden of learning every distinction from scratch. Neither pole is assumed here.

== 5.2. Thalamus-like routing and coordination

A thalamus-like motif draws attention to structured control over effective communication. Its characteristic bias is not necessarily the creation of rich content, but the regulation of which representations influence which components, with what gain, through which channel, and for how long.

This motif should not be reduced to a universal router. Biological thalamic organization suggests multiple nuclei embedded in different cortical and subcortical loops, which motivates factorized routing channels rather than one unrestricted switchboard. Different channels may regulate sensory flow, recurrent cortical interaction, memory access, or control-related communication while sharing only limited state.

The distinction from ordinary attention is one of emphasis rather than kind. Attention can implement a thalamus-like mechanism. The stronger idea is persistent, structured, and possibly low-bandwidth control of communication rather than unconstrained content mixing recomputed independently at every step. Routing state may itself be recurrent, allowing a temporary coalition or communication regime to remain active.

Open alternatives include whether routing acts on edges or recipient updates, whether it transmits content or only modulates another pathway, whether routes are competitive, and whether routing decisions are learned through the same signals as the content processors. It is also unclear when direct bypasses are necessary to prevent routing from becoming a brittle bottleneck.

== 5.3. Hippocampus-like binding and associative reinstatement

A hippocampus-like motif draws attention to rapid formation of separable conjunctive states, associative retrieval from partial cues, and reinstatement or replay of distributed information. The relevant distinction is not memory versus no memory. It is the contrast between preserving a particular conjunction and gradually extracting structure shared across many occurrences.

A conjunctive representation may bind context, entities, relations, temporal position, spatial information, and the states of several other modules. Sparse or high-dimensional coding may help separate similar conjunctions and reduce accidental overlap. A partial cue may later reactivate a larger configuration, either because the memory contains that configuration directly or because it helps reconstruct distributed state elsewhere.

“Index,” “episode,” and “replay” should remain flexible terms. A memory state need not be an arbitrary address, correspond to autobiographical experience, or store a literal snapshot. It may contain relational or predictive structure of its own. Replay may mean exact reinstatement, generative reconstruction, selective reactivation, or repeated influence on slower-learning systems.

Rapid binding also does not determine the storage substrate. Persistent activation, associative recurrent state, writable slots, temporary synaptic variables, fast weights, or mixtures of these could realize related functions. Important unresolved issues include allocation, capacity, interference, overwriting, novelty sensitivity, retrieval control, and the boundary between event-specific and reusable representations.

== 5.4. Basal-ganglia-like selection and value-sensitive gating

A basal-ganglia-like motif draws attention to learned selection among competing internal or external operations. Here a policy is understood broadly as a context-conditioned decision about what the system should do with its own computation. Possible objects of selection include state updates, memory writes, retrieval, pathway opening, module recruitment, persistence, computation depth, output choice, and transitions between operating regimes.

This motif highlights a distinction between constructing candidate content and selecting which transformation is permitted or maintained. It need not imply that content and selection are fully separate or that one controller governs the whole architecture. Parallel partially independent loops may control different domains, such as memory, communication, persistent context, or external output.

Value-sensitive or temporal-difference learning is one possible bias because internal decisions can have delayed consequences. It is not a required definition of the motif. Selection may also be shaped by end-to-end gradients, local objectives, learned evaluators, intrinsic signals, or combinations of them. The source and meaning of value are themselves open: external reward, prediction improvement, successful retrieval, uncertainty reduction, or preservation of future options may create different control regimes and different failure modes.

Questions to retain include whether selection is discrete or continuous, stochastic or deterministic, competitive or compositional, and whether it gates content, plasticity, persistence, or communication. Another tension is whether an explicit selector genuinely improves organization or merely relocates the original credit-assignment problem.

== 5.5. Cerebellum-like prediction, timing, and correction

A cerebellum-like motif draws attention to dense predictive learning, precise temporal structure, rapid error-driven adaptation, and correction of transformations implemented elsewhere. Several interpretations are relevant and should not be collapsed prematurely.

One interpretation is forward prediction of future sensory, latent, or output states. Another is residual correction: learning systematic errors in a slower or more general recurrent process and returning a compensatory signal. A third is inverse or control-oriented prediction of transformations that would produce a desired state. A fourth emphasizes timing, sequencing, and the reliable execution of frequently repeated dynamics.

A world-model-like objective may participate in such a motif, but the cerebellum-like component need not be the system's complete world model. Its distinctive tendency may instead be frequent local error, comparatively rapid adaptation, expansion into a representation suited to prediction, or a close coupling between predicted and realized transitions.

The substrate remains open. The component may be recurrent or predominantly feed-forward; its prediction may target observations, latent state, routes, outputs, or errors; and its correction may be additive, modulatory, or used only as a learning signal. It may complement flexible context-rich processing, or repeated interaction may cause the distinction between predictor and main process to blur.

== 5.6. Neuromodulatory control of operating regimes

A neuromodulatory motif draws attention to relatively low-dimensional signals that alter how other components process or learn rather than carrying detailed representational content. Candidate effects include changes in gain, update rate, plasticity, persistence, exploration, noise, routing priority, memory-write tendency, and sensitivity to particular inputs or errors.

Such signals may coordinate transitions between operating regimes: stable reuse and rapid adaptation, routine processing and high-compute deliberation, confident exploitation and exploration, or ordinary learning and heightened encoding of surprising events. Different modulatory channels may relate to novelty, uncertainty, expected value, arousal, task phase, or resource demand without mapping cleanly onto individual biological transmitters.

This motif differs from thalamus-like routing in emphasis. Routing changes which representations communicate or influence an update; modulation changes the conditions under which processing, updating, or learning occurs. The distinction may be useful even if both are implemented with multiplicative signals.

The main cautions are excessive globality and semantic overloading. A single scalar controlling many unrelated processes may become unstable or act as an uninformative shortcut. Multiple scoped modulators, each coupled to selected loops, may better capture the intended heterogeneity.

== 5.7. Salience, conflict, and meta-control

A salience- or conflict-monitoring motif draws attention to detecting when the current processing regime may be insufficient. Relevant conditions include surprise, prediction error, conflict among alternatives, uncertainty, retrieval failure, novelty, or unusually high expected value of further computation.

Its output might alter routing, recruit additional modules, preserve multiple hypotheses, trigger retrieval, increase computation depth, or change modulatory state. The motif need not solve the underlying task. It concerns recognition that ordinary processing should continue, widen, interrupt, or reconfigure.

This role overlaps with prediction, value estimation, routing, and neuromodulation. That overlap is informative rather than necessarily problematic. One open question is whether salience is a distinct computation or an emergent agreement among several local signals. Another is whether a centralized conflict estimate would aid coordination or erase the domain-specific meaning of uncertainty and failure.

== 5.8. Sensory and input organization

Input pathways draw attention to the fact that observations are not already in a form suitable for general recurrent computation. Different modalities or streams may benefit from local structure, specialized temporal scales, restricted receptive fields, uncertainty preservation, and progressive integration. Feedback from association, prediction, routing, or control processes may shape representation construction rather than leaving input processing purely feed-forward.

The relevant bias is not permanent dedication to a biological sense. It is sensitivity to the statistics and causal structure of an input stream, together with controlled convergence into shared representations. Parallel pathways may preserve different aspects of the same input, such as identity, location, dynamics, confidence, or task relevance.

Questions include how early different streams should share a geometry, whether feedback should alter features or only their effective relevance, and how modality-specific state participates in more general loops without being flattened prematurely.

== 5.9. Motor and output organization

Output pathways draw attention to the difference between selecting an intention and realizing a signal under interface-specific constraints. In an artificial system, output may mean physical control, language, communication, tool use, symbolic emission, or an update sent to another process.

A structured output pathway may separate broad intention, intermediate organization, and low-level realization. The appropriate hierarchy may vary substantially by domain. Basal-ganglia-like loops may influence selection, cerebellum-like processes may predict or correct consequences, and sensory feedback may close the recurrent loop.

This motif should not imply that internal computation culminates in one privileged motor stream. Internal routing, memory, and state changes can also be understood as actions. The useful distinction is between content or control that remains internal and transformations constrained by an external interface.

= 6. Loops, channels, and interaction patterns

The motifs above become most meaningful through interaction. A flat inventory of areas risks reproducing the very monolithic interpretation the framework is intended to avoid. Cross-area loops may be more fundamental units of organization than the areas themselves.

A memory-control loop, for example, could involve general representational modules, a conjunctive binding process, a selection mechanism controlling write or retrieval, and a routing channel that determines where reinstated information becomes effective. This description does not imply a selected architecture. It illustrates that “memory” may be a property of coordinated operations rather than one memory node.

A predictive-control loop could connect context-rich recurrent state, a fast predictor or corrector, a selector that determines whether a correction should affect ongoing dynamics, and modulatory signals controlling adaptation. A sensory-attention loop could combine modality-specific processing, routing, prediction, and conflict signals. Similar motifs may recur across several domains while maintaining mostly separate local states.

Several interaction properties deserve continued attention:

- *Factorization:* apparently global connections may consist of parallel channels with limited cross-talk.
- *Directionality:* feed-forward content, feedback context, error, and modulation may follow different paths.
- *Persistence:* a loop's communication or control regime may outlast the content that initially established it.
- *Privacy:* modules may preserve state that is not broadcast, exposing only selected summaries or requests.
- *Competition and cooperation:* loops may inhibit, recruit, or temporarily form coalitions with one another.
- *Shared resources:* distinct loops may reuse processors, memory, or modulatory signals without sharing their entire state.
- *Plasticity of topology:* some connectivity may be fixed, some softly gated, and some reorganized over longer timescales.

These properties help distinguish structured recurrent organization from a fully connected collection of named modules. At the same time, too much isolation can prevent integration, composition, and unexpected reuse. The intended space lies between unrestricted broadcast and rigid encapsulation.

= 7. Conceptual tensions to preserve

The following tensions are not defects to eliminate immediately. They identify places where the idea may develop in different directions.

== 7.1. Generality versus specialization

Weak biases should make some computations easier without fixing semantic roles. Yet a bias too weak to influence learning may be irrelevant, while a strong bias may hard-code the decomposition. The useful boundary may differ by dimension: connectivity can be strongly constrained while representational meaning remains open, or state geometry can be biased while routing remains flexible.

== 7.2. Areas versus loops

Area labels make the space cognitively manageable, but functions may belong primarily to recurrent loops. Treating areas as primary risks localization; treating loops as primary risks losing reusable local motifs. Both descriptions may be needed at different scales.

== 7.3. Content versus control

Routing, value, salience, and modulation are often described as control signals distinct from content. In a learned system the boundary may be fluid: a representation can influence routing, and a control state can carry task-relevant content. The distinction may still be valuable as a bias even if it is not ontologically clean.

== 7.4. Persistent state versus plasticity

Fast adaptation can occur through activation, writable memory, temporary synaptic change, or ordinary parameter updates. These substrates differ in capacity, stability, interference, and differentiability, but the conceptual roles do not dictate one choice. “Fast memory” should not silently become “fast weights.”

== 7.5. Local signals versus global coherence

Local predictive, value-related, or modulatory signals may improve organization and temporal credit, but they can also optimize incompatible proxies. A single global objective preserves coherence but may poorly shape internal decisions. The relationship among local and global learning is a central open dimension, not merely a technical detail.

== 7.6. Sparse communication versus integration

Restricted communication may protect specialization and reduce interference, but broad integration supports composition and flexible reuse. Learned routing does not automatically solve the tension because the router itself must obtain information and credit. Persistent channels, summaries, bypasses, and shared workspaces represent different possible compromises.

== 7.7. Biological guidance versus artificial opportunity

Biology reflects evolutionary, physical, metabolic, and developmental constraints that need not apply to artificial systems. Conversely, artificial systems have mechanisms unavailable to biology. The analogy is useful when it reveals a computational distinction or organization principle; it should be abandoned or transformed when it becomes a reason to preserve irrelevant details.

== 7.8. Named motifs versus independent biases

The neuroanatomical names make bundles memorable, but they can conceal the fact that connectivity, representation, update, timescale, control, and learning signals are independently variable. A future architecture may combine a hippocampus-like state geometry with thalamus-like access control and cerebellum-like predictive learning in one subsystem. Whether that is one area, several areas, or a loop should remain open.

= 8. Guidance for continued exploration

Further work using this document should deepen the conceptual map before narrowing it. Useful contributions include:

- identifying an overlooked computational motif or biological distinction;
- separating mechanisms that have been bundled under one area label;
- showing that two apparently different motifs are alternate views of the same bias;
- clarifying interactions, incompatibilities, or circular dependencies among loops;
- comparing multiple artificial realizations without treating the easiest one as the default;
- examining where a biological analogy illuminates the design space and where it becomes misleading;
- refining vocabulary so that architectural, dynamical, representational, and learning claims remain distinct;
- preserving questions whose answers would materially change the shape of the idea.

Unless a later task explicitly changes the mode, collaborators and AI agents should avoid responding to this context with a single recommended architecture, a staged implementation roadmap, a benchmark suite, or a claim that one minimal experiment captures the idea. Concrete mechanisms may be discussed as examples, but they should be presented alongside alternatives and the assumptions that make them relevant.

The current anchor is modest: complex recurrent computation may benefit from heterogeneous, partially specialized processes embedded in structured, overlapping loops. Neuroanatomy provides a rich source of candidate biases for imagining such organization. The point of the present document is to keep that space broad enough to discover what the idea is, while making it structured enough to think about together.
