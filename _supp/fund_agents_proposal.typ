#set page(
  width: 600pt,
  margin: (x: 8mm, y: 6mm),
)
#set text(
//   font: "Inter",
  size: 12pt
)

#let strength(body) = text(fill: rgb("#3b9a2c"), weight: "bold", body)
#let weakness(body) = text(fill: rgb("#b22222"), weight: "bold", body)

= Proposal: Fundamental RL Agents — Algorithms and Architectures

*Umbrella Vision.* We aim to develop *fundamental reinforcement learning (RL) agents* --- agents whose algorithms and architectures are *unified, general-purpose, and biologically inspired*. Current RL research often expands classical RL in fragmented ways, producing narrow solutions that do not combine well. Our direction is to move beyond these patchwork fixes and create *more general and compatible foundations*, while keeping backpropagation available as a tool when useful.

This serves two purposes:
- *Supportive* --- provide more flexible and fast pacing testbeds for biologically inspired cognitive models.
- *Practical* --- unify diverse RL subfields into broader, more fundamental approaches that scale and generalize.

== Research Directions

=== 1. Rich Implicit Reward / Cueing Systems

RL traditionally relies on a single scalar reward. In practice, researchers “pollute” this reward with auxiliary signals (intrinsic motivation, reward shaping), which mixes orthogonal objectives and requires fragile balancing. Instead, we propose learning a *cueing/conditioning system* alongside the extrinsic reward. This system is multidimensional, acts as a *modulator rather than a reward*, and is learned via the same extrinsic reward [over longer cross-episode horizon] conditioned on various supportive metrics (prediction error, policy entropy, etc.).

#underline[Why It Matters]
- Provides a principled alternative to hand-crafted intrinsic rewards and reward shaping.
- Avoids interference of orthogonal behaviors.
- Makes agents more controllable: external modulation of cues can adjust exploration, exploitation, risk aversion, or safety behavior.
- Encourages agents with richer internal states, while also improving explainability of their behavior and decision processes.
- Potentially better aligns with biological interpretations of conditioning: dopamine bursts tied to cues, not vanishing RPEs (reward prediction or TD error).

#underline[Examples / Entry Points]
- Study agents where exploration emerges from extrinsic-reward-trained cues.
- Investigate connections with Successor Features, curriculum learning, and safe RL.
- Explore cue-based interfaces for steering agent behavior externally.

=== 2. Multi-Horizon (Multi-$gamma$) Generalization

Extend policies and value functions to depend explicitly on $gamma$ --- $pi(dot, gamma)$ and $V(dot, gamma)$, --- enabling agents to *adapt dynamically to different time horizons*.

#underline[Why It Matters]
- Provides a complementary capability that can support other directions (lifelong and curriculum learning, exploration strategies, etc.).
- Bridges theory and practice by unifying Differential RL with popular reward normalization techniques (EMA + std-scaling).

#underline[Examples / Entry Points]
- Implement dynamic $gamma$-conditioned agents.
- Explore the connection between reward normalization and Differential RL in practice.

_(Note: this is likely less central than other directions, but it complements them and may naturally emerge as we push toward more general agents.)_

=== 3. Observation / Action Space Unification

Seek a minimal set of *generic sensorimotor primitives* that allow broad task coverage. Like humans achieve versatility with a handful of senses and actuators, agents might achieve generality with a small, reusable action/observation API.

#underline[Why It Matters]
- Enables large-scale, lifelong training across environments without per-environment re-engineering.
- Creates a common foundation for multimodal RL.

#underline[Examples / Entry Points]
- Propose primitives such as keyboard-like action spaces or visual-text input channels.
- Test feasibility in tractable setups like active vision and reading.
- Study whether shared primitives support transfer across domains.

=== 4. Active-X: Vision, Reading, Decision-Making

Perception becomes an *active traversal process*, not passive encoding. Agents actively choose *where and how to look, read, or think*, much like biological agents control their eyes or attention.

#underline[Why It Matters]
- Extends embodiment with richer motorics, improving generalization.
- Provides a unified way to process diverse data types (images, video, text, rendered text).
- Pushes RL agents toward active decision making: instead of acting at every time step, the agent can delay responses, take multiple internal "thinking" steps, and decide when to act.

#underline[Examples / Entry Points]
- Reformulate image classification and other CV tasks (segmentation, tracking) as active RL tasks: classify or segment using minimal glimpses.
- Develop unified multi-task setups for vision, reading, and multimodal inputs.

=== 5. Fundamental Multimodal RNN Architectures

_(I proposed this idea 2 months ago in mattermost' airi channel as "A Modular RNN Architecture with Inter-Column Attention")_

Move beyond the single-RNN backbone by building a *grid RNN* ($N$ RNNs, $H$ layers each) *that communicate via inter-column attention-like links* in addition to regular forward/upward paths. This creates large, factorized state spaces where some pathways are bound (inputs/outputs, aux losses in the spirit of predictive coding) while others remain unbound for emergent specialization.

#underline[Why It Matters]
- Serves as the architectural umbrella connecting other directions:
- Rich internal states (supporting cue systems).
- Multiple input/output channels (multimodality).
- Memory, recurrence and bound pathways (supporting model-based RL and intrinsic signal processing).
- Unlike transformers, these don’t rely on unlimited past access, but instead build dynamic state structures with specialization potential.

#underline[Starting steps]
+ Non-RL and RL memory tasks (test emergent memory storing specialization).
+ Model-based RL (use bound components for prediction, enforcing representation learning).
+ Lifelong multi-task RL (broad task distributions push the architecture toward useful specialization).


== Closing Note

Each direction tackles a different "axis of unification":
- Rewards $arrow.long$ cues and internal modulators.
- Horizons $arrow.long$ $gamma$-conditioning.
- Actions $arrow.long$ shared primitives.
- Perception/decisions $arrow.long$ active traversal and thinking.
- Architectures $arrow.long$ multimodal RNN grids tying the rest together.

Together, these form a coherent agenda toward fundamental RL agents with richer states, more general applicability, and deeper links to biology.