# Project Journey: rl_on_wheels

Detailed engineering notes covering the full evolution of the project:
SAC → TD3 → BC + DAgger → AIRL. Written as interview-prep reference —
every transition, every issue, every fix.

## Contents

1. [Task and goals](#1-task-and-goals)
2. [System architecture](#2-system-architecture)
3. [Phase 1: SAC (early baseline)](#3-phase-1-sac-early-baseline)
4. [Phase 2: TD3 (canonical model-free baseline)](#4-phase-2-td3-canonical-model-free-baseline)
5. [Phase 3: Behavior Cloning + DAgger](#5-phase-3-behavior-cloning--dagger)
6. [Phase 4: Inverse RL (AIRL)](#6-phase-4-inverse-rl-airl)
7. [Final comparison and what we learned](#7-final-comparison-and-what-we-learned)
8. [Q&A: likely interview questions](#8-qa-likely-interview-questions)

---

## 1. Task and goals

### What the robot does

A TurtleBot3 Waffle Pi navigates from a random spawn pose to a random
goal pose inside a walled arena. The arena contains static walls plus
dynamic cylindrical obstacles that move along scripted keyframe paths.
The robot has only **2D LiDAR** (360 rays → 36 normalised bins) as its
sensor — no camera. It outputs continuous **(linear velocity, angular
velocity)** commands at 10 Hz.

Episode ends when:

- robot reaches goal (within `goal_tolerance` metres) → **success**
- lidar reads a value below `collision_threshold` → **collision** (terminal)
- `max_episode_steps` is exceeded → **timeout** (truncation)

### Why this task

It's the standard testbed for mobile-robot navigation research: realistic
sensors (real lidar), continuous control, partial observability (lidar
can't see *through* dynamic obstacles' future positions), goal-conditioned
behaviour. Same difficulty class as Habitat / iGibson / PointGoal
challenges, but built on the open ROS2 + Ignition stack instead of a
custom physics layer.

### Project goal

Compare three families of approaches under matched conditions:

- **Model-free RL** (SAC, TD3, PPO)
- **Imitation Learning** (BC + DAgger from a Nav2 expert)
- **Inverse RL** (AIRL to recover the expert's reward, then optimise
  policy against it)

Headline number we want to know: which approach gives the **best policy
quality** per unit of compute, and what artifacts (reward function,
transferable knowledge) does each one produce.

---

## 2. System architecture

### Layered stack

```
┌─────────────────────────────────────────────────────┐
│ Python training scripts                             │
│   rl/train_td3.py, bc/train.py, airl/train_airl.py  │
└─────────────────────┬───────────────────────────────┘
                      │ gym.step(action) / gym.reset()
┌─────────────────────▼───────────────────────────────┐
│ TurtleBot3Env (gymnasium.Env)                       │
│   rl/envs/ros2_gym_env.py                           │
│   thin sync wrapper around ROS2 services            │
└─────────────────────┬───────────────────────────────┘
                      │ ROS2 service calls (sync RPC)
┌─────────────────────▼───────────────────────────────┐
│ env_bridge_node (C++)                               │
│   - assembles 42-dim observation                    │
│   - computes shaping reward + done                  │
│   - drives /cmd_vel into Ignition for step_duration │
└─────────────────────┬───────────────────────────────┘
                      │ /cmd_vel, /scan, /odom, /goal_pose
┌─────────────────────▼───────────────────────────────┐
│ Ignition Fortress (physics + lidar simulation)      │
│   waffle_pi robot model + DiffDrive plugin          │
│   gpu_lidar sensor (3.5 m or 6 m max)               │
│   dynamic_obstacle_node moves cylinders by keyframes│
│   reset_node teleports robot + spawns goal sphere   │
└─────────────────────────────────────────────────────┘
```

### Why this layering

- **Python at the top** because every modern RL library (SB3, imitation,
  gymnasium) speaks gym.Env. Keeping the env wrapper thin means we can
  swap algorithm libraries without touching the sim.
- **C++ env_bridge_node** because the observation assembly and reward
  computation need to run in lock-step with the sim's physics tick.
  Doing this in Python would cost ROS2 IPC latency per field of the
  observation and miss the 10 Hz deadline.
- **ROS2 services (not topics)** between gym wrapper and bridge because
  the gym contract is *synchronous*: `step(action)` returns the resulting
  obs/reward/done together. Topics would create races.
- **Ignition Fortress** rather than Gazebo Classic because Fortress is
  the actively-supported branch with native ROS2 bridge support.

### Observation vector (42-dim)

```
[lidar_0 … lidar_35]    36 × normalised LiDAR ∈ [0, 1]    (robot frame)
[dist_norm]              distance to goal / max_lidar_range
[cos_goal_body]          cos of goal heading in robot frame
[sin_goal_body]          sin of goal heading in robot frame
[goal_path_min]          min LiDAR in ±20° cone around the goal direction
[prev_lin_vel]           previous step's linear  velocity
[prev_ang_vel]           previous step's angular velocity
```

Design notes:

- **Lidar bin layout**: 36 bins, each covering 10° of the 360° fan. Bin 0
  faces forward, indices increase counter-clockwise.
- **Goal in body frame, not world frame**: makes the policy invariant to
  absolute world position. The policy sees "goal is 0.4 to my left," not
  "goal is at world (2.1, 1.3)."
- **`goal_path_min`** is a critical engineered feature. It asks "is the
  straight path to the goal blocked?" by min-pooling the 5 lidar bins
  closest to the goal angle. Cheaper than learning this from raw lidar.
- **`prev_lin_vel`, `prev_ang_vel`** included for RL (helps the value
  function reason about momentum) but **dropped for BC** to avoid an
  action-echo shortcut (see Phase 3 issues).

### Action space

```
linear_vel  ∈ [0.0,  0.22]   m/s    (no reverse)
angular_vel ∈ [-2.0, 2.0]    rad/s
```

Forward-only matches typical mobile-robot constraints (differential
drive can pivot but doesn't have a reverse gear that we use). The 0.22
m/s and 2.0 rad/s caps are the manufacturer-spec maxes for Waffle Pi.

### Stages

The arena layout varies per stage. Stages 1-4 are 5×5 m; stage 5 is
7×7 m (introduced for the IRL phase to test generalisation).

| Stage | Arena | Inner walls | Dynamic obstacles | Used for |
| ----- | ----- | ----------- | ----------------- | -------- |
| 1     | 5×5   | no          | 0                 | empty-arena RL warm-up |
| 2     | 5×5   | no          | 4 static          | static-obstacle test |
| 3     | 5×5   | no          | 4 small oscillation | gentle dynamic |
| 4     | 5×5   | 7 segments  | 2 keyframe        | **canonical RL + BC training stage** |
| 5     | 7×7   | 8 segments  | 6 keyframe        | **generalisation probe / IRL phase** |

### Reward function (used by RL phases)

```
reward = r_yaw + r_vangular + r_vlinear + r_distance + r_obstacle − 1

terminal: +2500 on goal_reached, −2000 on collision
```

| Term | Range | Purpose |
| ---- | ----- | ------- |
| `r_yaw = -abs(goal_angle)` | [−π, 0] | face the goal |
| `r_vangular = -ω²` | [−4, 0] | discourage spinning |
| `r_vlinear = -((v_max − v) × 10)²` | [−5, 0] | encourage forward motion |
| `r_distance = 2·d₀/(d₀ + d) − 1` | [−1, 1] | shape progress toward goal |
| `r_obstacle = -20 if min_obs_dist < 0.22m else 0` | {0, −20} | avoid moving cylinders |

The `-1` constant is a time penalty per step (prefers shorter episodes).

Why dense shaping rather than sparse goal-only reward: with random
initial policies, the robot would never stumble on a goal in a 5×5 m
arena. Dense shaping gives gradient information *everywhere*. The
terminal `+2500` / `−2000` are large enough to dominate the cumulative
shaping, so the policy is ultimately optimised for actual success, not
just for the shaping signal.

---

## 3. Phase 1: SAC (early baseline)

### Why we started with SAC

For continuous-action robot control, the natural first choice is one of
the modern off-policy actor-critic methods: SAC, TD3, or DDPG. SAC was
the first one we set up because:

- **Soft actor-critic with entropy regularisation** is well-known to be
  more robust to hyperparameter choices than TD3/DDPG. Good starting
  point when you don't yet know what numbers to tune.
- **Stochastic policy** explores naturally; you don't need to engineer
  an OU noise schedule like DDPG/TD3.
- **Automatic temperature tuning** (`ent_coef: "auto"` in SB3) means
  the entropy/exploitation trade-off is learned, not hand-set.

The agent is in `rl/agents/sac_her.py` (despite the filename, it's
**plain SAC** — there was a HER experiment early on but it's not used
in the current code path). Config is `configs/sac_her.yaml`.

### Key hyperparameters chosen

```yaml
sac:
  learning_rate: 3.0e-4    # SB3 default
  buffer_size: 1_000_000   # full episode history
  batch_size: 256
  tau: 0.005               # soft target update
  gamma: 0.99
  train_freq: 1            # one update per env step
  gradient_steps: 4        # see note below
  ent_coef: "auto"
  target_entropy: "auto"   # = -action_dim = -2
  learning_starts: 5000    # collect random data before training
  policy_kwargs: net_arch=[512, 512]
```

### Issue: Q-value divergence at high gradient_steps

**Symptom**: setting `gradient_steps: 8` made the Q-function diverge
within the first 50 k env steps. The critic loss would blow up, the
actor would track that, and the policy collapsed to a near-constant
action.

**Root cause**: gradient_steps > 1 means we run multiple critic
updates per single environment step. With 8 grad steps per env step
and a replay buffer that's only growing at 10 Hz of env data, the
critic was overfitting to the most recent (tiny) slice of buffer faster
than fresh diversity could arrive. The bootstrap target moved faster
than the data distribution changed.

**Fix**: `gradient_steps: 4` is a safe compromise — still extracts
more signal per env step than the default 1 (which is wasteful on a
slow ROS sim), but doesn't overfit. The yaml comment documents this
trade-off explicitly for posterity.

### Why we eventually moved off SAC to TD3

Honest answer: **simplicity and reproducibility**. SAC has more moving
parts (entropy tuning, stochastic actor sampling, the temperature loss),
which made debugging the training curve harder. TD3 is more boring —
deterministic actor, two Q-functions, target policy smoothing, that's
it. For a project where the sim is the bottleneck and we want to spend
debugging effort on the *environment* (reward shaping, curriculum,
stage construction), TD3's fewer-knobs profile was the right call.

**SAC was not rerun on the curriculum stages** — it was the initial
prototype before the stage system existed. All curriculum experiments
in this repo use TD3.

---

## 4. Phase 2: TD3 (canonical model-free baseline)

### Why TD3

Once we had decided on TD3 (see "why we moved off SAC" above), the
specific reasons we liked it for our task:

- **Deterministic policy** = simpler to evaluate. `model.predict(obs,
  deterministic=True)` just returns the actor's output; no
  re-sampling, no entropy-regularised value head to interpret.
- **Twin critics + delayed policy update** = robust to Q-value
  overestimation, which we'd just been burned by in SAC.
- **Off-policy** = replay buffer means we don't waste hard-earned
  trajectories. Important on a slow sim.
- **OU exploration noise** = controllable. We can schedule the
  exploration → exploitation transition explicitly.

Agent: `rl/agents/td3_agent.py`. Config: `configs/td3.yaml`. Train script:
`rl/train_td3.py`.

### Key hyperparameters

```yaml
td3:
  learning_rate: 3.0e-4 → 1.0e-4    # see "plateau" below
  buffer_size: 1_000_000
  batch_size: 128
  tau: 0.003
  gamma: 0.99
  policy_delay: 2
  target_policy_noise: 0.2
  target_noise_clip: 0.5
  net_arch: [512, 512]
  action_noise:
    type: "ou"
    sigma: 0.1 → 0.05    # see "exploration" below
    theta: 0.15
  frame_stack: 1
  action_repeat: 4    # smoother motion
```

### Issue: TD3 plateau at ~59 % success

**Symptom**: on the canonical training stage (stage 4, 5×5 maze + 2
dynamic cylinders), TD3 climbed steadily to ~59 % success rate by
around 200 k env steps, then sat there for tens of thousands of steps
without further improvement. The TB curves looked converged but the
number was stuck.

**Root cause**: the policy had found a *local optimum* — a behaviour
that's safe enough to be rewarded by `r_distance + r_yaw` for most
episodes but doesn't handle the harder dynamic-obstacle cases. The
3e-4 learning rate, which was right for the *initial* learning phase,
was now perturbing the policy too aggressively in fine-tune territory.
Exploration noise was also too high; the policy kept getting kicked
around its local optimum.

**Fix** (recorded in the yaml comments):

1. **Drop LR 3e-4 → 1e-4**: smaller gradient steps preserve the
   converged policy while allowing slow refinement.
2. **Reduce OU noise sigma 0.1 → 0.05**: less perturbation around the
   current policy. The intuition is *anneal exploration*: high early
   when the policy is bad, low later when the policy is good.

Result: pushed past the plateau to higher success rates.

### Issue: `action_repeat=4` for smoother control

**Symptom**: at the default 10 Hz step rate and `max_angular_vel = 2.0
rad/s`, the robot's behaviour was juddery — the policy would output a
hard left command for one step, then a hard right command the next.
Physically realistic robots can't change direction at 10 Hz cleanly;
the DiffDrive plugin was straining.

**Fix**: `action_repeat = 4` means each policy action is applied for
4 sim steps before the policy is queried again. Effective control
frequency drops to 2.5 Hz. Smoother motion, less physics stress, also
4× more sim-time per policy decision → effectively faster training
progress per env step.

### The bigger problem: TD3 was memorising, not learning to navigate

The plateau itself was a numerical symptom; the deeper concern was
**what the policy had actually learned**. Looking at TD3's rollouts
qualitatively:

- The policy did well on its *training stage* (stage 4) — that's
  where the 59 % number came from.
- But the *trajectories looked like memorisation*. The robot was
  taking very specific paths consistent with the specific wall
  positions of stage 4, not paths that responded to lidar geometry
  in general. If you watched the rollout it almost looked like the
  robot was following a hand-drawn route, not "see obstacle → turn
  away."
- This is the classic **state-space overfitting** failure mode of
  RL trained on a fixed environment: the policy learns the
  *specific* `(absolute_lidar_pattern, action)` mapping that's
  rewarded, rather than the *general* `(obstacle-direction, turn-
  away)` rule that would transfer.

The practical evidence: when I evaluated the TD3 checkpoint on stages
with the same physics + similar but not identical wall layouts, the
success rate dropped sharply. The policy was *stage-specific*, not a
general navigator. Reward shaping had pushed the policy into a local
optimum where it could collect distance-reward + face-goal-reward on
the trained layout, without developing the underlying obstacle-
avoidance competence that would let it generalise.

This is *the* well-known problem with RL on fixed-environment
benchmarks — and it's the reason the field has moved toward
domain-randomised training and curriculum learning. We had a
curriculum (stages 1-4 of increasing difficulty) but in practice each
stage was a separate training run, and the policy that came out of
stage-4 training didn't transfer cleanly to slightly-different layouts.

### Why we moved off TD3 to imitation learning

Three reasons crystallised:

1. **TD3 was overfitting to the layout, not learning navigation.**
   See above. Even if we'd pushed past the 59 % plateau on stage 4,
   we'd have a stage-4-specialist, not a generalist navigator.
2. **Nav2 was already sitting in our ROS2 workspace.** It's a
   production-grade mobile-robot navigator with global path planning
   (NavFn over a costmap), local control (Regulated Pure Pursuit),
   and behaviour-tree recovery logic. Crucially, **Nav2's behaviour
   is fundamentally generalisable** — it's running classical
   planning algorithms over a costmap, not memorised state-action
   pairs. A policy that imitates Nav2 should inherit at least some
   of that generalisability, since the demos themselves come from a
   general planner reasoning about the local geometry rather than
   the specific layout.
3. **Sim throughput was the binding constraint.** RL papers train on
   millions of env steps with vectorised parallel sims. We had a
   single ROS sim at 10 Hz. Continuing to push TD3 was diminishing
   returns; we'd take 50+ hours to maybe gain another 10 pp on stage
   4, and we'd still have a stage-4-specialist at the end.

So we pivoted: use Nav2 as a *teacher*, do supervised imitation from
its trajectories. That's Phase 3.

One honest caveat worth knowing: BC has its own overfitting failure
mode (covariate shift — the policy fails when it visits states the
expert never showed it). We addressed that with DAgger in Phase 3 and
saw the same generalisation hypothesis bear out qualitatively — the
BC + DAgger policy navigated through the dynamic-obstacle stages with
behaviour that *looked* like "see obstacle → turn away," not "follow
a specific path." The 84 % stage 4 result with much shorter mean
episode lengths than TD3 backs this up: BC was finding direct routes
to the goal, not memorised long routes.

---

## 5. Phase 3: Behavior Cloning + DAgger

### The pivot

The insight: **for navigation tasks, a hand-engineered classical
planner (Nav2) is a much stronger demonstrator than an RL policy is at
this compute budget**. So we record Nav2 driving the robot, learn to
imitate it.

This is on the `curriculum` branch.

### Pipeline

```
1. Sim + Nav2 launched   tb3_rl_bridge bridges sim to ROS2.
                         tb3_nav2 launches Nav2 lifecycle nodes.
2. Episode driver        bc/run_episodes.py calls /reset_episode →
                         pose_publisher publishes map→odom TF →
                         goal_forwarder sends goal to /navigate_to_pose.
3. Bag recording         ros2 bag captures /scan /odom /goal_pose
                         /robot_world_pose /cmd_vel during episodes.
4. Dataset build         bc/build_dataset.py walks the bag, pairs
                         each /cmd_vel with the most recent obs,
                         keeps only successful episodes → NPZ.
5. Supervised train      bc/train.py — MLP[512,512] + tanh head,
                         MSE on rescaled actions, frame stacking
                         (4 frames default), Adam.
6. Eval                  bc/eval.py — deterministic rollout with
                         safety guards (speed cap, front + side
                         clearance, near-goal P-controller).
```

### New ROS2 package: tb3_nav2

Custom nodes:

- **`pose_publisher_node.cpp`**: subscribes to Ignition's ground-truth
  pose feed (`/world/empty/dynamic_pose/info`) and computes
  `T_map_odom = T_map_robot · T_odom_robot⁻¹`, broadcasts it as the
  `map → odom` TF. This is the **localisation shim** — in a real robot
  we'd use AMCL or particle filters here, but in sim we can use
  ground-truth.
- **`goal_forwarder_node.cpp`**: subscribes to `/goal_pose` (which
  `reset_node` publishes at episode reset) and forwards it to Nav2's
  `/navigate_to_pose` action server. Also republishes Nav2's result
  (succeeded / aborted / canceled) as a `Bool` on `/nav_episode_result`
  so the orchestrator can detect episode end.

### Issue: stale goal_forwarder results

**Symptom**: the `run_episodes.py` orchestrator was seeing rapid
~390 ms "aborts" between episodes, treating them as failures and
moving on. But Nav2 had only just been *told* about the new goal — it
couldn't have actually attempted and aborted that fast.

**Root cause**: the previous episode's Nav2 goal was being preempted
when the new goal arrived. The Nav2 action server fires a `CANCELED`
result for the preempted goal *after* `async_cancel_goal()`. That
result-callback would arrive at the orchestrator *labelled as if it
belonged to the current episode*, because the goal_forwarder was
using a single `current_handle_` that pointed to whichever goal was
most recent.

**Fix**: in `goal_forwarder_node.cpp`, introduce a `goal_counter_`
that increments on every new goal. The result callback captures the
counter value at the time it was registered (call it `my_id`) and
only publishes its result if `my_id == goal_counter_` — i.e., if no
newer goal has arrived in the meantime. Stale results are silently
dropped.

This was the trickiest debugging session in the BC pipeline because
the failure mode looked like "Nav2 is broken" when it was actually
"the bookkeeping around preemption was racing."

### Issue: env_bridge tracking robot in wrong frame

**Symptom**: BC eval succeeded geometrically (robot was right next to
goal sphere) but env reported failure (distance > goal_tolerance).

**Root cause**: the C++ env_bridge tracked robot position from `/odom`
(which is in the `odom` frame), but the goal was published in the
`map` (world) frame. `/odom` drifts relative to `map` over long
episodes. The distance computation `||robot_odom − goal_map||` mixes
frames.

**Fix**: added `/robot_world_pose` (PoseStamped in `map` frame)
published by `pose_publisher_node` from Ignition's ground-truth.
env_bridge subscribes to that instead of `/odom`. The frame mismatch
is removed at the source.

### Issue: goal_tolerance mismatch BC eval ↔ Nav2 demos

**Symptom**: BC was visually reaching the goal but eval marked it
false.

**Root cause**: Nav2's `xy_goal_tolerance` is 0.40 m — Nav2 considers
itself "arrived" at 0.40 m and stops. The demos therefore contain
trajectories that *only* get as close as ~0.30-0.40 m. BC learned to
stop in that range. But `env_bridge_node` had `goal_tolerance: 0.20 m`
— it wouldn't mark success until 0.20 m. There's a dead zone of 0.10-
0.20 m where BC is visually done but env says no.

**Fix**: bumped env's `goal_tolerance` to **0.40 m** in
`bridge.launch.py` to match Nav2's. Honest threshold — demos didn't
contain examples of "drive closer than 0.40 m," so BC was never going
to learn that.

### Issue: BC action-echo feedback loop (40 vs 42 dims)

**Symptom**: BC trained well (MSE dropping nicely) but evaluated at
**0 % success** on initial runs. Visually the robot was just executing
the same action over and over without responding to obstacles.

**Root cause**: the env's 42-dim observation includes `prev_lin_vel`
and `prev_ang_vel`. During training, the action for step `t` is highly
correlated with the action for step `t−1` (smooth navigation). So
the policy learned the *shortcut*: "if `prev_lin_vel` is high, output
high `lin_vel` too." It effectively echoed the previous action,
ignoring the actual situation. At eval time, this echo became a
self-reinforcing loop and the policy stopped responding to the world.

**Fix**: drop `prev_lin_vel` and `prev_ang_vel` from BC's observation.
BC sees the first 40 dims only (`obs[:40]`). The action-echo shortcut
becomes impossible; the policy is forced to use lidar + goal info.

**This was the most important single fix in the BC pipeline.** It
took the policy from 0 % to a meaningful baseline (~30 %).

### Issue: frame stacking for dynamic obstacles

**Symptom**: even after fixing the action-echo, BC failed on episodes
where a dynamic cylinder crossed in front. The lidar at frame `t` and
the lidar at frame `t-1` look different, but the *single-frame* policy
sees only frame `t` and can't tell whether the obstacle is moving
toward or away.

**Fix**: **frame stacking**. The policy reads 4 consecutive observations
concatenated (4 × 40 = 160 dims). The policy can now implicitly compute
"this lidar bin dropped from 0.8 to 0.6 to 0.4 over the last 3 frames →
something is approaching" via standard MLP computation.

Implementation:

- `bc/train.py` stacks frames within each episode at load time (using
  `ep_id` to find episode boundaries).
- `bc/eval.py` maintains a `deque(maxlen=frame_stack)` at runtime and
  feeds the concatenation to the policy.
- `frame_stack=4` saved in the checkpoint so eval can reproduce the
  same layout.

Result: stage 4 BC baseline 28 % → DAgger iter 2 84 % → frame-stacked
80 % (slight raw drop but much more honest collision avoidance).

### DAgger: closing the covariate-shift gap

**The covariate-shift problem**: BC only ever sees states that the
expert (Nav2) visited. If BC's deployed policy ever drifts into a state
the expert never visited, BC has no training data for that situation
and might output garbage.

**DAgger's solution**: iteratively, have BC drive the robot. At every
state BC visits, ask the expert "what would you do here?" and add
`(state, expert_action)` to the training set. Retrain BC. Repeat.

**Shadow-mode Nav2**: the technical trick that lets us run "BC drives,
Nav2 watches." In `tb3_nav2/launch/nav2_bringup.launch.py`, when the
`dagger_mode:=true` argument is passed, the controller_server node has
its `cmd_vel` topic remapped to `cmd_vel_expert`. Nav2 still computes
its actions but they go to a sink topic; BC's policy publishes the
*real* `/cmd_vel`. The DAgger collector subscribes to
`/cmd_vel_expert` to capture Nav2's "what would I have done" signal.

### Issue: stale expert pairs (the ros2 spin problem)

**Symptom**: first DAgger collection run kept only 57 pairs out of
5167 candidates — 98.9 % stale rate. The `expert_fresh()` check was
finding the most recent Nav2 cmd_vel message was old enough to discard.

**Root cause**: the expert listener was a `rclpy.Node` subscribing to
`/cmd_vel_expert`. We were calling `rclpy.spin_once(expert,
timeout_sec=0.01)` once per env step to drain queued messages. But
`env.step()` blocks on a ROS2 service round-trip (~100 ms+), during
which Nav2 publishes ~1 message that queues in the listener. The
`spin_once(0.01)` after step didn't reliably process that queued
message before we read the listener's `t_last`. Effectively the
listener was constantly stale.

**Fix**: put the expert listener on a **background `SingleThreadedExecutor`
spinning on its own thread**. Now expert callbacks fire continuously
in the background regardless of what `env.step()` is doing on the main
thread. Stale rate dropped from 98.9 % to ~37 %, and the remaining
37 % is genuine Nav2 silence during recovery behaviours (which we
*should* drop — they're not useful labels).

### Issue: train/eval distribution mismatch in DAgger

**Symptom**: first valid DAgger iteration on stage 4 *decreased*
success rate (28 % → 25 %). The labels looked fine in distribution
analysis but BC was getting worse.

**Root cause**: `bc/eval.py` applies a hard front-clearance safety
override (`obs[39] < 0.08 → kill linear, force turn`) that keeps BC
out of the worst states. But `bc/dagger_collect.py` did *not* apply
that override — BC drove without safety, ended up in states the
override would have prevented, and those states got labelled with
Nav2's actions. So BC was training on labels for states it would
never visit at deploy time.

**Fix**: make `dagger_collect.py` apply the *same* safety stack as
`eval.py`. Now the states BC visits during collection match the states
it'll visit at deploy time. Distributions match → DAgger labels are
useful again.

Result on stage 4: DAgger iter 1 → 50 % → iter 2 → 84 %.

### Stage 5: the 7×7 generalisation arena

Built specifically for the IRL phase: bigger arena, 8 inner walls
spread out instead of 7 clustered, 6 dynamic cylinders instead of 2,
outer walls scaled from ±2.425 to ±3.425 m.

### Issue: lidar/arena ratio collapse on stage 5

**Symptom**: BC trained on stage 5 demos at the default 3.5 m lidar
range plateaued at 32-42 % (noisy). DAgger iter 1 actively hurt the
policy (dropped to ~10 %).

**Root cause**: in a 5×5 arena, 3.5 m lidar covers ~70 % of the diagonal
— BC can see most of the world. In a 7×7 arena, 3.5 m covers only ~50 %
— BC has *less context per observation* than it did on stage 4. Same
network, harder problem.

**Fix**: bump the lidar hardware max from 3.5 m to 6.0 m on stage 5.

- SDF: `<max>3.5</max> → <max>6.0</max>` in `tb3_stage5.sdf`'s lidar sensor.
- Launch: `max_lidar_range` parameter becomes stage-conditional via
  PythonExpression in `bridge.launch.py` (6.0 only for stage 5, 3.5
  otherwise).
- All scaling that uses `max_lidar_range` (normalisation in env_bridge,
  build_dataset, eval safety thresholds) auto-adjusts.

### Issue: scaling safety thresholds for stage 5

**Symptom**: after the lidar bump to 6.0 m, the eval-time safety
overrides started firing constantly. Robot was paralysed.

**Root cause**: the overrides used **normalised lidar values** as
thresholds. `front_clearance < 0.13` means "0.13 of the max range."
At 3.5 m lidar that's 0.45 m physical. At 6.0 m lidar that's *0.78 m*
physical — way too generous, fires whenever any wall is within a
metre.

**Fix**: rewrite the override in `bc/eval.py` to use *physical* metres
as the threshold:

```python
max_range = float(cfg["env"]["max_lidar_range"])
front_threshold = 0.45 / max_range    # physical 0.45 m, normalised
```

Now the threshold autoscales correctly with `max_lidar_range`. Stage 4
still gets 0.13 (= 0.45/3.5); stage 5 gets 0.075 (= 0.45/6.0).

### Issue: side-wall scraping (the visually-obvious one)

**Symptom**: user observation: "the robot is sliding over the wall.
It doesn't trigger a collision but you can see the gap is zero."

**Root cause**: the original front-clearance override only fires when
obstacles are in the **goal cone** (`obs[39]`, ±20° around goal
direction). A wall directly to the *side* of the robot doesn't enter
the goal cone — override doesn't fire — BC happily cruises along the
wall scraping it. The env's `collision_threshold` (0.13 m) is barely
above the lidar grazing distance so it counts as success.

**Fix**: add a **second safety override** that checks the front 180°
(lidar bins 0-9 and 27-35 = front-left and front-right 90°).

```python
front_left  = obs_bc[:10]
front_right = obs_bc[27:36]
if min(front_left.min(), front_right.min()) < side_threshold:
    action[0] = 0.0
    action[1] = -1.5 if min(front_left) < min(front_right) else +1.5
```

Side threshold tuned tighter than front (0.25 m vs 0.45 m) because
side walls in narrow corridors are unavoidable but front walls aren't.

### Issue: BC stopping just short of the goal

**Symptom**: BC consistently parks ~0.50 m from the goal and never
crosses the 0.40 m success threshold. Episodes time out.

**Root cause**: Nav2's demos all *end* with full stop at the 0.40 m
threshold. BC learned "near goal → slow down" but never saw demos of
"actually cross the success threshold." It parks at the edge.

**Fix**: a **final-approach P-controller** override. When robot is
within 0.7 m of the goal AND no safety guard is firing, take over BC's
output:

```python
goal_angle = atan2(obs[38], obs[37])
action[0] = 0.10                          # crawl
action[1] = clip(1.5 * goal_angle, ±1.0)  # P on heading
```

This is the textbook proportional controller. It bypasses BC for the
last 0.7 m of approach and reliably closes the gap.

### Final BC + DAgger results (50-ep deterministic eval)

**Stage 4 (5×5 maze, 2 dynamic cylinders)**:

| Stage of training | Success | Mean reward | Mean len |
| ----------------- | ------- | ----------- | -------- |
| Base BC | 28 % | −1271 | 264 |
| + DAgger iter 1 | 50-60 % | — | — |
| + DAgger iter 2 | **84 %** | +1326 | 204 |
| + Frame stacking | 80 % | +1207 | 168 |

**Stage 5 (7×7 maze, 6 dynamic cylinders)**:

| Config | Success | Mean reward | Mean len |
| ------ | ------- | ----------- | -------- |
| 3.5 m lidar, base BC | 32-42 % | −1102 | 244 |
| 6 m lidar + scaled safety + P-controller | **62 %** | +154 | 200 |

---

## 6. Phase 4: Inverse RL (AIRL)

### What we wanted from IRL

Two goals:

1. **Recover an interpretable reward function** that captures what
   Nav2 was implicitly optimising. BC produces only a policy; you
   can't introspect "why does the policy do X here?" IRL recovers a
   reward signal you can inspect, transfer to other environments,
   or combine with hand-designed objectives.
2. **Try to exceed the expert.** In theory, an IRL-recovered reward
   could be optimised by an RL agent to find a policy *better* than
   the demonstrator. The hope was for a policy that approaches or
   exceeds Nav2's 89 % on stage 5.

This is on the `irl` branch.

### Why AIRL specifically (vs GAIL, SQIL, etc.)

- **AIRL** = Adversarial Inverse RL. Same adversarial setup as GAIL
  but the discriminator is structured so you can extract a clean
  reward function: `r(s,a,s') = g(s,a) + γh(s') − h(s)`. The `g`
  term is the "true reward" component, free of shaping. This is
  what `inspect_reward.py` probes.
- **GAIL** does not factor reward this way — you'd get only a
  policy, no inspectable reward.
- **SQIL** is simpler but also doesn't produce a reward function.
- **MaxEnt-IRL** with neural reward is theoretically clean but
  notoriously slow.

For the "we want an inspectable reward" goal, AIRL is the right
choice.

### Implementation stack

- **`imitation` library** (HumanCompatibleAI) — has `AIRL` and
  `BC` algorithms that integrate with SB3.
- **SB3 PPO** as the policy generator (AIRL needs an on-policy
  RL algorithm; the imitation library doesn't support off-policy
  generators).
- **`BasicShapedRewardNet`** with `RunningNorm` input normalisation
  for the reward function.

Files in `airl/`:

| File | Purpose |
| ---- | ------- |
| `convert_demos.py` | BC NPZ → `imitation.data.types.Trajectory` pickle |
| `train_airl.py` | BC pretraining + AIRL adversarial loop |
| `train_with_reward.py` | Phase 2 — PPO against frozen reward |
| `inspect_reward.py` | Sweep recovered reward over obs axes → PNG plots |
| `eval_policy.py` | 50-ep deterministic eval for SB3 PPO checkpoints |

### Conceptual AIRL loop (what happens each round)

1. **Robot rolls out in sim**. PPO drives for `n_steps = 1024` env
   steps (~100 s of sim time). Records `(s, a, s')` triples.
2. **Discriminator sees two piles**: real Nav2 triples (from
   demos) vs PPO-rollout triples. Trains a binary classifier to
   distinguish them.
3. **Discriminator output → reward**: for each transition in PPO's
   rollout, the discriminator's "this looks like Nav2" probability
   becomes the reward.
4. **PPO trains on those rewards**. Actions that "look like Nav2"
   are reinforced; actions that don't are suppressed.

Loop. The reward function (the `g` head of the reward net) silently
absorbs everything the discriminator has learned — by end of training
it encodes the pattern distinguishing "Nav2 navigation" from
"not-quite-Nav2 navigation."

### Issue: variable-horizon rejection

**Symptom**: first training run crashed almost immediately:

```
ValueError: Episodes of different length detected: {450, 332, 350, 31}.
Variable horizon environments are discouraged.
```

**Root cause**: the `imitation` library refuses variable-length
episodes by default. The reasoning is that episode length can *leak
reward information* — the agent can infer "if my episode is short, I
probably terminated badly." This can let the discriminator find
shortcut features.

**Fix**: pass `allow_variable_horizon=True` to the `AIRL` constructor.
For navigation our episodes have a legitimate task-induced length
distribution (goal reached at different times, occasional collisions,
occasional timeouts). The library's safety rail doesn't apply.

### Issue: reward curve descending monotonically (first run)

**Symptom**: the AIRL reward seen by PPO (`raw/gen/rollout/
ep_rew_wrapped_mean` in TensorBoard) was a smooth monotonic descent
from −150 to −710 over 48 k steps. Policy at end of run evaluated to
**16 % success**.

**Root cause**: this is the canonical "discriminator dominates" AIRL
failure. PPO started from a random initialisation. Random PPO in a
sim with dynamic obstacles spins/crashes in seconds. Nav2 demos are
long, purposeful trajectories. The discriminator's job (distinguish
them) became trivial: 100 % accuracy on day one. So every action PPO
took got a uniformly large negative reward — no useful gradient
information for the policy to climb out of the hole.

**Fix attempt 1**: lower `n_disc_updates_per_round` 4 → 1. Idea:
slow down the discriminator so PPO can keep up. *Still descending.*
The discriminator was so easy to train that 1 update per round was
still enough.

**Fix attempt 2 (worked)**: **BC pretraining**. Use the
`imitation.algorithms.bc.BC` class to train PPO's policy network
directly on the demos for 20 epochs, *before* AIRL starts. Now PPO
begins close to Nav2's behaviour distribution; the discriminator
can't trivially separate them, and PPO has real gradient to work
with.

```python
bc_trainer = BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=rollout.flatten_trajectories(trajectories),
    policy=policy.policy,   # share weights with the PPO actor
    rng=np.random.default_rng(args.seed),
)
bc_trainer.train(n_epochs=20)
```

Result: with BC warm-start, the reward curve became a healthy
U-shape — dip in the first 5 k steps as discriminator catches up,
then climb back. Final policy reaches 46 %.

### Issue: PPO destroying BC quality during AIRL refinement

**Symptom**: even with BC warm-start, the env-eval callback at
round 10 was showing 0 % success — BC's good initial policy got
*wiped out* by the first few PPO updates in AIRL.

**Root cause**: PPO's default learning rate (3e-4) was tuned for
training-from-scratch. Applied to a BC-pretrained policy, a single
PPO update at 3e-4 was enough to drag the policy off the good BC
manifold. The discriminator then trained against a *degraded* policy,
and the recovered reward got polluted.

**Fix**: drop PPO LR 10× to **3e-5**. Same BC warm-start, but PPO's
updates are now conservative enough that BC quality survives the
first AIRL rounds. The discriminator gets to learn against a *good*
policy, recovered reward is cleaner.

Round 10 eval after the fix: **60 %** (vs 0 % with default LR). The
fix worked.

### Issue: tightening exploration noise around BC

**Symptom**: after BC pretraining, the PPO policy mean is close to
Nav2's behaviour, but PPO samples actions from a Gaussian with mean
= BC output and std = 1.0 (default). With action bounds of [0, 0.22]
for linear velocity, std = 1.0 is *enormous noise* — every action is
basically random.

**Fix**: after BC training, manually set the policy's `log_std`
parameter:

```python
with torch.no_grad():
    policy.policy.log_std.data.fill_(-1.5)  # std ≈ 0.22
```

Now the action distribution is tight around the BC mean. Exploration
still happens (PPO does sample), but the samples stay close to
sensible navigation actions.

### Issue: env-success ≠ AIRL reward

**Symptom**: phase 1 final eval (after the LR + log_std fix): 46 %
success on 50 episodes. Within noise of BC's 52 %.

**Diagnosis**: AIRL's reward (what PPO maximises) and env success
rate are **not aligned**. PPO can keep increasing the discriminator
reward without that translating into real navigation improvement.
In fact the policy can chase adversarial shortcuts that score high
on the discriminator but fail on actual goals.

**Fix**: env-eval-based checkpointing. The `train_airl.py` script
now has a callback that runs *real environment evals* every N AIRL
rounds (default every 10), and saves the policy snapshot whenever
env success improves. The final `policy_best_eval.zip` is whichever
mid-training checkpoint had the highest *real* success rate, not
whichever scored highest on the discriminator.

This prevents "I trained too long and the policy drifted past its
peak" — a known AIRL failure mode.

### Phase 2: PPO against frozen reward (the textbook IRL workflow)

The classic IRL recipe is:

> Step A: recover the reward function from demos.
> Step B: freeze the reward. Train a fresh RL agent against it.

Step B is the chance for IRL to actually *exceed* the expert —
because the RL agent isn't bound by the demos' specific trajectories,
only by what the reward says is good.

We implemented this as `airl/train_with_reward.py`. Key idea:

```python
class FrozenRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, _env_r, term, trunc, info = self.env.step(action)
        with torch.no_grad():
            r = self.reward_net(
                last_obs, action, obs, done=0
            ).item()
        return obs, r, term, trunc, info
```

The env now returns the AIRL reward instead of the hand-shaped env
reward. PPO trains on stationary rewards (no more adversarial
co-evolution). BC warm-start preserved.

### Result of phase 2: 50 % success

Within noise of BC (52 %) and of AIRL adversarial (46 %). The
**recovered reward function works as a signal** — PPO can train
against it without diverging — but the policy doesn't exceed BC.

Why this is the structural ceiling: the recovered reward is
fundamentally an *imitation reward*. It rewards "looks like Nav2."
Optimising it converges to "looks like Nav2" ≈ "what BC produces."

### Issue: frame stacking hurts AIRL (the counter-intuitive one)

**Hypothesis going in**: frame stacking lifted BC from 52 % → 62 %
on stage 5. Apply the same trick to AIRL → should give similar
+10 pp.

**Result**: AIRL with frame_stack=4 collapsed to **16 % adversarial,
10 % phase 2**. Worse than no stacking.

**Root cause**: 4× larger input (40 → 160 dims) means the
discriminator and policy nets have 4× more first-layer parameters.
With our small demo set (47 k pairs) and short training budget
(60-200 k env steps), the larger networks **under-train**. The
discriminator finds noisy gradients faster than PPO can
compensate. BC supervised learning doesn't have this issue
because it gets to make 20 full passes over the data offline.

So: **frame stacking helps BC because BC is supervised; it hurts
AIRL because AIRL is adversarial + RL-bottlenecked**. A non-obvious
finding worth documenting.

### Final AIRL results (50-ep deterministic eval, stage 5)

| Variant | Success | Mean reward | Mean len |
| ------- | ------- | ----------- | -------- |
| AIRL adversarial (default LR=3e-4) | 16 % | −1608 | 188 |
| AIRL adversarial (LR=3e-5 + BC warm-start) | 46 % | −196 | 121 |
| **Phase 2 (PPO + frozen reward)** | **50 %** | **+17** | **111** |
| AIRL + frame stacking | 16 % | — | — |
| Phase 2 + frame stacking | 10 % | — | — |

### Recovered reward function inspection

`airl/inspect_reward.py` probes the trained reward net by sweeping
synthetic observations and measuring the reward. Three sweeps:

1. **Reward vs distance to goal** — confirms reward increases as
   dist-to-goal decreases (monotonic structure).
2. **Reward vs lidar clearance** — confirms reward decreases sharply
   as min lidar drops toward the collision threshold.
3. **Reward vs goal angle** — confirms reward peaks when goal is in
   front of the robot (cos goal angle = 1).

Plots committed at `airl/results/*.png`. These are the **IRL
contribution** — interpretable artifacts BC cannot produce.

### Honest IRL summary

- **Pipeline works end-to-end**: BC → AIRL adversarial → phase 2.
- **Recovered reward is interpretable** and shows the expected
  structure.
- **Policy quality matches but does not exceed BC** — 50 % vs 52 %.
- **The ceiling is structural**: pure imitation-based IRL is bounded
  by expert quality. Exceeding Nav2 would require either much more
  compute (vec envs, millions of steps) or a hybrid reward (AIRL
  reward + task reward), neither of which we did.

---

## 7. Final comparison and what we learned

### The headline table (stage 5, 50-ep eval)

| Method | Success | Notes |
| ------ | ------- | ----- |
| Nav2 expert | 89 % | the demo source — upper bound for imitation |
| **BC + frame stack + safety overrides** | **62 %** | strongest learned policy |
| BC alone (matched eval) | 52 % | imitation baseline |
| Phase 2 PPO + frozen AIRL reward | 50 % | matches BC |
| AIRL adversarial | 46 % | matches BC within noise |
| (TD3 not rerun on stage 5) | — | implemented for stages 4+ |

### Key technical takeaways

1. **On a slow single-sim training budget, BC is the strongest tool
   for policy quality**. BC's 30-second supervised training extracts
   more signal per unit time than thousands of env steps for
   RL/IRL.

2. **DAgger is the key to BC quality past covariate shift**. Pure BC
   plateaus around 30 %. DAgger pushed stage 4 to 84 %.

3. **Adversarial training is brittle**. AIRL needs BC warm-start AND
   conservative PPO learning rate AND env-eval-based checkpointing
   to even match (not exceed) BC.

4. **Frame stacking helps supervised methods but can hurt adversarial
   ones**. The signal/budget trade-off matters: bigger input = more
   to learn = more parameters that need data.

5. **Lidar/arena ratio matters for navigation policies**. The same
   lidar range gives a smaller fraction of the world in a bigger
   arena. Stage 5 needed 6 m lidar to match the implicit
   "observability fraction" of stage 4.

6. **The IRL contribution is the recovered reward function, not the
   policy**. For an inspectable model of "what does the expert care
   about," IRL has no equal. For raw policy quality, BC wins.

### What we'd do differently with more time

- **Vectorised environments** (multiple parallel sims). This is the
  single biggest unlock for the RL/IRL methods — would 4-8× the
  effective training throughput and might let AIRL actually exceed
  BC. Engineering cost: rewrite the ROS2 bridge to support multiple
  sim instances, or migrate to a faster sim like Isaac.
- **DAC (off-policy AIRL with SAC)**. ~2-5× sample efficient over PPO
  + AIRL. Would let us train longer in the same wall clock budget.
- **Hybrid reward (AIRL + task)**. Stops being "pure IRL" but is the
  most likely path to actually exceeding the expert. ~6 hr more
  training.
- **More demonstrations**. 47 k pairs is fine for BC but marginal for
  AIRL. Another 500 Nav2 episodes would let the discriminator fit
  more nuanced features.

---

## 8. Q&A: likely interview questions

### "Walk me through the project"

> The task is goal-conditioned navigation for a TurtleBot3 in ROS2 +
> Ignition Fortress, with only 2D lidar as sensor input. I implemented
> and compared three families of approaches end-to-end:
> first model-free RL (SAC then TD3 — SAC was the prototype, TD3
> became the canonical baseline because it has fewer moving parts);
> then behavior cloning + DAgger using Nav2 as the demonstrator (this
> ended up being the strongest method, 84 % success on the canonical
> training stage); then AIRL — inverse RL — to recover an interpretable
> reward function from the same demos.
>
> The headline finding: on this single-sim compute budget, BC +
> DAgger gives the best policy quality, while AIRL contributes an
> interpretable reward function as a different kind of artifact.
> AIRL doesn't exceed BC on raw policy quality, which is consistent
> with the broader literature on imitation-bounded methods.

### "Why did you start with SAC and then move to TD3?"

> SAC was the initial prototype — stochastic policy with automatic
> entropy tuning is robust to hyperparameter choices, so it's a good
> "first one to try" for continuous control. The pivot to TD3 was for
> simplicity and reproducibility: TD3 has fewer moving parts (no
> entropy term, no temperature loss, deterministic actor), which
> made debugging the training curves easier. For a project where the
> sim was the bottleneck and I wanted to spend debugging effort on
> the environment and reward design, TD3's fewer-knobs profile won.

### "Why did you move off TD3 to imitation learning?"

> Two reasons, and the second is the more important one. The first is
> the obvious one — TD3 plateaued at ~59 % on the canonical training
> stage even after dropping the LR and reducing exploration noise,
> and pushing further would have taken tens of hours on a single
> 10 Hz sim for diminishing returns.
>
> The more important reason is that TD3 was *overfitting to the
> layout*. When I watched the rollouts, the policy was taking very
> specific routes that matched the specific wall positions of stage
> 4, not paths that would respond to lidar geometry in general. It
> was a stage-4-specialist, not a general navigator. This is the
> classic state-space overfitting failure of RL on a fixed
> environment — the policy memorises `(absolute_lidar_pattern,
> action)` pairs that get rewarded, rather than the underlying
> `(obstacle-direction, turn-away)` rule. Evaluating on similar but
> non-identical layouts confirmed it — success dropped sharply.
>
> Imitating Nav2 was the right pivot because Nav2's behaviour is
> fundamentally generalisable — it runs classical path planning
> (NavFn over a costmap) plus a regulated pure pursuit controller,
> reasoning about local geometry rather than memorised state-action
> pairs. A policy that imitates Nav2 should inherit some of that
> generalisability since the demos themselves come from a general
> planner. The BC + DAgger policy on stage 4 (84 % with much shorter
> mean episode lengths than TD3) bears this out — it found direct
> routes to the goal, behaviour that looked like "see obstacle →
> turn away" rather than memorised long routes.

### "What was the hardest debugging session?"

> The 0 % BC eval. I had trained BC supervised, the loss curve looked
> healthy, the model checkpoint loaded fine, but at eval time the
> robot didn't react to its environment at all. Took a while to
> realise it was an action-echo feedback loop — the policy had
> latched onto `prev_lin_vel` and `prev_ang_vel` in the observation
> as a shortcut, effectively "if I was moving forward last step, keep
> moving forward." Fixed by dropping those two dims from the
> observation. Took the policy from 0 % to a real baseline in one
> change.

### "How does DAgger work in your setup?"

> Standard DAgger: BC drives, expert relabels every state.
> Implementation-wise I used a "shadow Nav2" trick — in the launch
> file, I remap the controller_server's `cmd_vel` topic to
> `cmd_vel_expert` when a `dagger_mode:=true` argument is set. So
> Nav2 still computes its actions but they go to a sink topic; my BC
> policy publishes the real `/cmd_vel`. The DAgger collector
> subscribes to `/cmd_vel_expert` to record what Nav2 would have done
> at each state BC visits.

### "What is AIRL and how does it differ from BC?"

> BC is supervised learning: train a network to copy expert actions.
> AIRL is inverse reinforcement learning: train a discriminator to
> distinguish expert from policy trajectories, use the discriminator's
> score as a reward signal, train a policy (PPO in my case) to
> maximise that reward. Whichever policy fools the discriminator best
> is the answer. The discriminator's internal structure also lets you
> extract an interpretable reward function, which is the main appeal —
> BC produces only a policy.

### "Why didn't AIRL exceed BC in your project?"

> Three reasons. First, compute budget — AIRL papers train on
> millions of env steps with vec envs; I had a single sim at 10 Hz.
> Second, adversarial training is brittle — needs many gradient
> updates to stabilise, which we couldn't afford. Third, structural:
> AIRL's reward is fundamentally an imitation reward (looks like
> expert), so optimising it caps at expert quality. Genuinely
> exceeding the expert requires either much more data/compute, or
> hybrid imitation + task reward, neither of which I did. The result
> I have — recovered reward function plus policy matching BC — is
> consistent with what most IRL papers report on hard real-ish
> tasks.

### "What's the difference between AIRL adversarial and phase 2?"

> Phase 1 of AIRL is the standard adversarial loop: discriminator
> and policy update simultaneously, reward is whatever the
> discriminator says at the time. This is unstable — I saw the
> reward curve descend monotonically in early runs because the
> discriminator dominated.
>
> Phase 2 is the textbook IRL recipe: once you've recovered the
> reward function, *freeze* it, and train a fresh RL agent against
> just that reward. No more adversarial drift. The reward is
> stationary, RL converges cleanly. In theory this is where IRL can
> exceed the expert because the RL agent isn't bound by specific
> demo trajectories. In my case it didn't exceed, just matched —
> 50 % vs BC's 52 % — because of the imitation ceiling.

### "How did you handle the slow sim?"

> Three approaches: (a) tuned `action_repeat=4` for TD3 so each
> policy decision covers 0.4 s of sim time instead of 0.1 s, giving
> 4× more effective training per env step. (b) For BC, the dominant
> training time is supervised (offline on the demo dataset) — sim
> is only needed for collection and eval, both of which are
> finite-budget. (c) For AIRL, I added env-eval-based checkpointing
> so even if training degrades past its peak, I keep the best
> checkpoint. This is essentially "early stopping on real metric"
> instead of trusting the AIRL reward.

### "Why didn't you use vec envs?"

> Honest answer: each parallel env would need its own ROS2 sim
> instance, which means separate Ignition processes, separate
> namespace setup, separate bridge nodes. The engineering work is
> significant — a few days at least. For this project I prioritised
> getting a clean comparison across three method families over
> optimising one of them with vec envs. With more time I'd build
> the parallel-sim infrastructure as the next step.

### "What did frame stacking change?"

> For BC: lifted stage 4 from 28 % → 80 % effectively (combined with
> DAgger) and stage 5 from ~42 % → 62 %. The reason: dynamic
> obstacles are invisible to a single-frame policy — it can't tell
> a static wall from a moving cylinder at the same lidar reading.
> Stacking 4 frames lets the network compute lidar deltas implicitly
> and infer motion.
>
> Counter-intuitively, frame stacking *hurt* AIRL — pulled it from
> 46 % down to 16 %. The reason: BC is supervised and gets 20 passes
> over the data, so it can fit a larger input. AIRL is adversarial
> + RL-bottlenecked, and the 4× larger input means more parameters
> to fit with the same training budget. Under-training of the
> discriminator becomes worse than its baseline behaviour. Documented
> in the project notes as a real finding.

### "What would you do if you had another month?"

> Probably split it three ways. First week: vec envs and migrate to
> a faster sim so the RL/IRL methods aren't compute-bound. Second
> week: re-run AIRL phase 2 with the larger budget and proper frame
> stacking — see if exceeding BC is actually achievable in this
> task. Third week: a hybrid reward variant — AIRL reward plus a
> small task reward signal — which is the realistic path to
> meaningfully exceeding BC on real tasks. Last week: writeup,
> result reproducibility, and a real-robot deployment test of the
> best BC + DAgger policy as the sim-to-real transfer experiment.
