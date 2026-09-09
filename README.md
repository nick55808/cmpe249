# Cmpe249 - Perception-Aware Teleoperation Data Quality for Imitation Learning
Created by: Nicholas Bao, Baron Cai, Sai Teja Nomula

## Abstract
Humanoid robots are increasingly trained via imitation learning on teleoperated demonstrations, and recent work confirms that demonstration *quality* — not just task success — is a major bottleneck: novice operators produce jerky, inconsistent, or otherwise flawed demonstrations that still "succeed" but poison downstream policy learning. The closest existing system for this problem (DQAF, 2026) scores real operator sessions holistically after the fact, but because real sessions confound many issues at once and have no ground-truth label for *why* a given demonstration is bad, it cannot say which specific factor — latency, camera noise, occlusion, reduced resolution, incorrect depth, or operator error — actually drives policy failure, or by how much. This project uses a simulator (Isaac Sim, or a fallback simulator if setup time proves prohibitive) to generate humanoid manipulation demonstrations with each of these six factors injected independently at controlled, graded severity levels, trains behavior-cloning policies per condition, and measures policy degradation factor-by-factor — producing, for the first time, a ranked, causal picture of which teleoperation-pipeline weaknesses matter most. A ROS 2 pipeline records, scores (rule-based scorer vs. a lightweight learned classifier/vision model), and labels demonstrations in real time on a laptop, and is evaluated against the simulator's ground-truth corruption labels rather than human judgment.

## Why this project

There is a consensus that data quality is the bottleneck in humanoid imitation learning (that premise is well supported — see `LITERATURE_SURVEY.md`), but no existing system isolates *which* quality dimension matters most, because real recorded teleoperation sessions can't cleanly separate "the camera was noisy" from "the operator was clumsy" from "there was 200ms of latency" — they all happen together, uncontrolled. Simulation is what makes the controlled version of this question answerable: it can inject exactly one factor at exactly one severity and hold everything else fixed, which is not possible with real operator recordings. That controlled, causal, per-factor design — rather than a holistic quality score — is this project's specific contribution.

## Alignment with the course-suggested direction

*Track: research*

This project sits closest to the "**VLA robotics**" example from CMPE 249 Lecture 1's "Example research project directions" slide: *"Fine-tune OpenVLA on a small manipulation task or simulated robot."* It reframes that direction from "fine-tune a policy for capability" to "diagnose what makes the training data for that policy good or bad" — same problem space (robot learning from demonstrations on a simulated manipulation task), different question. It also touches the "Safety and evaluation" suggested track, since a ranked, causal understanding of which data-collection failures matter most is directly a safety/reliability question for any team deploying a teleoperation data pipeline at scale.

## Scope

Given how fast a full 6-factor × multi-severity × multi-seed experimental grid grows, the core deliverable is scoped to a smaller subset — a small number of factors (e.g., latency, camera noise, occlusion) at two severity levels each, on one manipulation task — with the remaining factors and severities as a stretch goal once the pipeline is validated end-to-end. "Operator mistakes" is treated separately from the other five sensor-pipeline corruptions, since it requires its own operational definition (scripted deliberate errors vs. an actual human operator instructed to err) before it can be injected in a controlled way (see Novelty & Feasibility Audit).

