Literature & SOTA Survey

Organized around why demonstration quality matters, what the closest existing system does (and doesn't) diagnose, and the infrastructure this project's ablation study builds on. Papers are all from the last 1–3 years (with one classic exception, DART, kept because it is the direct methodological ancestor of this project's design), per the assignment requirement.

Theme 1 — Teleoperation demonstration quality (closest prior work — defines the gap)
Closing the Loop in Teleoperation: Episode-Level Data Quality Assessment and Feedback for High-Quality Demonstration Collection (DQAF) — arXiv:2605.26349, 2026. The closest existing system: scores real novice-operator teleoperation sessions on motion smoothness, joint saturation, stalling, and gripper chatter, and shows task success and demonstration quality are not the same thing. Critically, it is observational and holistic — it scores whole sessions after the fact and cannot isolate which specific factor (latency vs. noise vs. occlusion vs. operator error) caused a low score, because real sessions confound all of these at once and provide no ground-truth label for the cause. This project's controlled, per-factor simulation design is the direct answer to that limitation.
Theme 2 — Why demonstration quality matters causally (imitation-learning theory)
Feedback in Imitation Learning: The Three Regimes of Covariate Shift — arXiv:2102.02872. Explains the mechanism: behavior-cloning policies only learn states seen in training demonstrations, so small deployment-time errors compound because the policy drifts into unseen states — this is the theoretical reason why which demonstrations you collect, not just how many, determines downstream policy reliability.
DART: Noise Injection for Robust Imitation Learning — Laskey et al., CoRL 2017 — PMLR PDF. The direct methodological ancestor of this project: deliberately injects noise into demonstrations to study/improve policy robustness. This project extends the same "controlled injection" idea from generic noise to a structured comparison across six named, teleoperation-specific corruption factors, each at graded severity — DART does not decompose noise into named causes and compare their relative effect.
Theme 3 — Teleoperation-specific latency and sensing degradation (defines which factors to simulate)
A Latency-Aware Framework for Visuomotor Policy Learning on Industrial Robots — arXiv:2602.14255v1. Shows control-loop latency specifically degrades visuomotor policy learning; motivates latency as one of the controlled factors.
A Learning-Driven Visual Servoing Framework for Latency Compensation in Image-Guided Teleoperation — ScienceDirect, 2026. Further evidence that latency compensation is an active, unresolved problem in image-guided teleoperation specifically (not just generic robot control).
Theme 4 — Simulation infrastructure for humanoid demonstration generation
NVIDIA Isaac Sim: Enabling Scalable, GPU-Accelerated Simulation for Robotics — arXiv:2606.03551. The simulator this project's data-generation pipeline is built on; documents the sensor-physics and synthetic-data-generation capabilities needed to inject controlled camera noise, occlusion, resolution, and depth-error conditions.
OASIS: From Simulation Data Collection to Real-World Humanoid Loco-Manipulation — arXiv:2606.08548. A recent example of a simulation-based humanoid demonstration-collection pipeline; used as a feasibility reference for how much simulation infrastructure is realistic to stand up in a semester.
Theme 5 — ROS 2 real-time execution (relevant to the System/Deployment components)
A Survey of Real-Time Support, Analysis, and Advancements in ROS 2 — arXiv:2601.10722. What ROS 2's executor model does and doesn't guarantee timing-wise — relevant since the quality-monitor pipeline needs to score and label demonstrations in real time within a ROS 2 node.
Timing Analysis and Priority-driven Enhancements of ROS 2 Multi-threaded Executors — arXiv:2408.08440. Shows ROS 2's own executor scheduling is itself a source of timing variance — relevant context for keeping the "real-time" claim about the laptop-based quality monitor honest and scoped.
Where the gap is

DQAF establishes that demonstration quality is a real, measurable bottleneck, and the covariate-shift theory explains mechanistically why quality (not just quantity) drives policy reliability — but neither isolates which teleoperation-pipeline factor is responsible, because real recorded sessions confound latency, sensing noise, occlusion, and operator skill simultaneously with no ground-truth label separating them. DART shows controlled noise injection is a viable experimental design, but only for generic, undifferentiated noise, not a comparison across named, realistic corruption sources. Nobody has run a controlled, per-factor, graded-severity ablation — using simulation's ability to inject exactly one factor at a time and hold everything else fixed — to produce a ranked, causal answer to "which specific teleoperation weakness should a team fix first." That ranked, causal picture is this project's specific, previously unmeasured contribution.

Rewrite README.md for BEV corruption×compression topic
Rewrite LITERATURE_SURVEY.md
Rewrite NOVELTY_AUDIT.md
Rewrite build_proposal.js and regenerate PROJECT_PROPOSAL.docx
Send updated files and draft team talking points
Rewrite README.md for humanoid teleoperation data-quality ablation
Rewrite LITERATURE_SURVEY.md for teleoperation data-quality topic
Rewrite NOVELTY_AUDIT.md for teleoperation data-quality topic
Rewrite build_proposal.js and regenerate docx
Send updated deliverables
LITERATURE_SURVEY.md
README.md
PROJECT_PROPOSAL.docx
NOVELTY_AUDIT.md
Connectors
Web search
Skills
Create a skill for this kind of task
