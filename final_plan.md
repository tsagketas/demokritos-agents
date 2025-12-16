# 🎯 Project D: Fictitious Play & Reinforcement Learning
**Learning equilibria in repeated zero-sum games**

---

## 📋 Quick Overview

**Goal:** Implement FP and RL agents that learn to play repeated zero-sum games, compare convergence, stability, and performance.

**Team:** 2 persons  
**Split:** 30% theory / 70% implementation  
**Timeline:** Dec 20 → Feb 15

---

## 🗓️ Timeline at a Glance

| Week | Dates | Phase | Deliverable |
|------|-------|-------|-------------|
| 0 | Dec 20 | Topic declaration | ✓ Email sent |
| 1 | Dec 20-27 | Theory + Setup | Git repo, notes |
| 2 | Dec 28 - Jan 3 | Game framework | Working games |
| 3 | Jan 4-10 | Agents | FP + RL working |
| 4 | Jan 11-17 | Experiments | Results + plots |
| 5 | Jan 18-24 | Analysis | Draft report |
| 6 | Jan 25-31 | Prep meeting | Demo ready |
| - | **Jan 27-31** | **1st Meeting** | Feedback |
| 7 | Feb 1-7 | Refinement | Final results |
| 8 | Feb 8-14 | Finalization | Presentation |
| - | **Feb 15** | **Final Presentation** | 🎯 |

---

## 📁 Project Structure
```
project/
│
├── games/
│   ├── base_game.py              # Abstract game class
│   ├── matching_pennies.py       # 2x2 zero-sum
│   └── rps.py                    # Rock-Paper-Scissors
│
├── agents/
│   ├── base_agent.py             # Agent interface
│   ├── fictitious_play.py        # FP implementation
│   └── q_learning.py             # Q-learning agent
│
├── experiments/
│   ├── runner.py                 # Experiment engine
│   ├── fp_vs_fp.py              # FP vs FP matchup
│   ├── rl_vs_rl.py              # RL vs RL matchup
│   └── fp_vs_rl.py              # Cross-play
│
├── analysis/
│   ├── metrics.py                # Distance to Nash, exploitability
│   └── visualizer.py             # Plotting utilities
│
├── results/                      # Saved plots & data
│   ├── plots/
│   └── data/
│
├── report/                       # LaTeX/Markdown
│   ├── main.tex
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🎮 Games to Implement

### Game 1: Matching Pennies (MUST) ⭐
- **Size:** 2x2
- **Type:** Pure zero-sum
- **Nash:** (0.5, 0.5) - uniform mixed
- **Why:** Simple, clear convergence

**Payoff Matrix:**
```
        H    T
    H [ 1  -1]
    T [-1   1]
```

### Game 2: Rock-Paper-Scissors (MUST) ⭐
- **Size:** 3x3
- **Type:** Cyclic zero-sum
- **Nash:** (1/3, 1/3, 1/3)
- **Why:** Non-trivial dynamics, oscillations

**Payoff Matrix:**
```
        R    P    S
    R [ 0   -1    1]
    P [ 1    0   -1]
    S [-1    1    0]
```

### Game 3: Stochastic Game (OPTIONAL) 🎁
- State-dependent payoffs
- Only if time permits
- **Not required for good grade**

---

## 🤖 Algorithms to Implement

### A. Fictitious Play

**Core Idea:**
1. Track opponent's action history
2. Compute empirical distribution (belief)
3. Play best response to belief

**Pseudocode:**
```python
class FictitiousPlayAgent:
    def __init__(self):
        self.opponent_history = []
        self.belief = uniform_distribution()
    
    def act(self, game):
        return best_response(self.belief, game)
    
    def update(self, opponent_action):
        self.opponent_history.append(opponent_action)
        self.belief = empirical_dist(self.opponent_history)
```

**Properties:**
- ✅ Guaranteed convergence in 2-player zero-sum
- ✅ Simple to implement
- ⚠️ Can be exploited early on

---

### B. Q-Learning

**Core Idea:**
1. Learn action values Q(a)
2. ε-greedy exploration
3. Update Q based on rewards

**Pseudocode:**
```python
class QLearningAgent:
    def __init__(self, n_actions, lr=0.1, epsilon=0.1):
        self.Q = np.zeros(n_actions)
        self.lr = lr
        self.epsilon = epsilon
    
    def act(self):
        if random() < self.epsilon:
            return random_action()
        return argmax(self.Q)
    
    def update(self, action, reward):
        self.Q[action] += self.lr * (reward - self.Q[action])
```

**Hyperparameters to test:**
- Learning rate: `[0.01, 0.1, 0.5]`
- Epsilon: `[0.05, 0.1, 0.2]`

---

## 🧪 Experiments

### Setup

**3 Core Matchups:**
1. **FP vs FP** - Theoretical convergence
2. **RL vs RL** - Self-play learning
3. **FP vs RL** - Cross-play dynamics

**Parameters:**
- Iterations: 10,000
- Runs per config: 10 (for averaging)
- Games: Matching Pennies + RPS

---

### Metrics to Track

| Metric | Description | Formula |
|--------|-------------|---------|
| **Distance to Nash** | How far from equilibrium | `\|\|strategy - Nash\|\|₂` |
| **Exploitability** | Max gain from best response | `max_a E[u(a, strategy)]` |
| **Cumulative Reward** | Total payoff over time | `Σ rewards` |
| **Regret** | Loss vs optimal play | `Σ(best - actual)` |
| **Variance** | Strategy stability | `σ²(strategy)` |
| **Convergence Speed** | Iterations to reach 95% Nash | `min t: \|\|strategy(t) - Nash\|\| < 0.05` |
| **Strategy Stability** | Variance in mixed strategy over time | `σ²(strategy_t)` per iteration |

### Enhanced Evaluation (Beyond Payoff)

**Robustness Testing:**
- FP vs RL vs Random opponent
- Adaptation analysis when opponent changes
- Failure scenarios: when FP/RL fails and why

**Comparative Analysis:**
- Why FP works well or fails in specific scenarios
- Why RL adapts better in certain situations
- Multi-agent learning insights

---

### Plots to Generate (minimum 8)

#### Must-Have:
1. **Strategy Evolution** (line plot)
   - X: iterations, Y: probability per action
   
2. **Distance to Nash** (line plot)
   - X: iterations, Y: L2 distance
   
3. **Cumulative Reward** (line plot)
   - Compare FP vs RL
   
4. **Exploitability Heatmap** (for RL)
   - X: learning rate, Y: epsilon, Color: final exploitability
   
5. **Average Payoff** (bar chart)
   - Per agent type per game

#### Nice-to-Have:
6. **Phase Portrait** (for RPS)
   - 3D simplex showing trajectory
   
7. **Convergence Speed** (bar chart)
   - Iterations to reach 95% Nash
   
8. **Regret Curves** (line plot)
   - Cumulative regret over time

---

## 📊 Expected Results

### Matching Pennies
- **FP vs FP:** Fast convergence to (0.5, 0.5)
- **RL vs RL:** Slower, but reaches equilibrium
- **FP vs RL:** RL exploits FP early, FP adapts

### Rock-Paper-Scissors
- **FP vs FP:** Oscillations before convergence
- **RL vs RL:** High variance, eventual convergence
- **FP vs RL:** More stable than RL vs RL

---

## 📝 Report Structure (12-15 pages)

### 1. Introduction (1 page)
- Problem statement
- Motivation: why compare FP and RL?
- Contributions

### 2. Background (2-3 pages)
**2.1 Zero-Sum Games**
- Definition
- Nash equilibria
- Mixed strategies

**2.2 Fictitious Play**
- Algorithm
- Convergence theorem (cite Robinson 1951)
- Best response computation

**2.3 Q-Learning**
- Algorithm
- Exploration vs exploitation
- Convergence properties

### 3. Implementation (2 pages)
**3.1 Game Specifications**
- Payoff matrices
- Nash equilibria

**3.2 Agent Architectures**
- FP: belief updates
- RL: Q-value updates

**3.3 Hyperparameters**
- Learning rates, epsilon, iterations

### 4. Experimental Setup (1 page)
- Matchups
- Metrics
- Hardware/software specs

### 5. Results (4-5 pages)
**5.1 Matching Pennies**
- FP results + plots
- RL results + plots
- Comparison

**5.2 Rock-Paper-Scissors**
- Same structure

**5.3 Cross-Play Analysis**
- FP vs RL dynamics

### 6. Discussion (1-2 pages)
**Key Insights:**
- FP converges fast but exploitable early
- RL robust but needs tuning
- Learning rate critical for RL
- Exploration-exploitation tradeoff

**Enhanced Analysis:**
- **Failure Scenarios:** When does FP fail? When does RL fail? Why?
- **Adaptation Analysis:** How do agents adapt when opponent changes?
- **Robustness:** FP vs RL vs Random opponent performance
- **Practical Implications:** What does this mean for real-world multi-agent systems?
- **Multi-agent Learning Insights:** Deeper understanding beyond "who wins more"

### 7. Conclusions (0.5 page)
- Summary
- Future work

### References
- Shoham & Leyton-Brown textbook
- Robinson 1951 (FP convergence)
- Watkins 1989 (Q-learning)
- 5-10 papers total

---

## 🎤 Presentation (15-18 slides)

### Slide Breakdown

1. **Title Slide**
   - Project name, team, date

2. **Motivation** (1 slide)
   - "How do rational agents learn in games?"

3. **Problem Definition** (1 slide)
   - Repeated zero-sum games
   - Nash equilibria

4. **Games** (1 slide)
   - Show payoff matrices

5. **Fictitious Play** (2 slides)
   - Slide 1: Algorithm + intuition
   - Slide 2: Example step-by-step

6. **Q-Learning** (2 slides)
   - Slide 1: Algorithm + intuition
   - Slide 2: Exploration strategy

7. **Experimental Setup** (1 slide)
   - Matchups, iterations, metrics

8. **Results: FP vs FP** (2 slides)
   - Matching Pennies + RPS
   - Strategy evolution plots

9. **Results: RL vs RL** (2 slides)
   - Reward curves
   - Sensitivity analysis (lr/epsilon)

10. **Results: FP vs RL** (2 slides)
    - Cross-play dynamics
    - Exploitability analysis

11. **Key Takeaways** (1 slide)
    - **FP:** Fast convergence, brittle
    - **RL:** Robust, needs tuning
    - **Trade-off:** Speed vs stability

12. **Conclusions** (1 slide)
    - Summary + future work

---

## 👥 Work Distribution

### Person 1: "Theory & FP"
**Responsibilities:**
- Game implementations (`matching_pennies.py`, `rps.py`)
- Fictitious Play agent
- FP experiments (FP vs FP)
- Theoretical sections in report (Background 2.1, 2.2)
- Presentation slides 1-5

**Estimated hours:** 40-50

---

### Person 2: "RL & Experiments"
**Responsibilities:**
- Metrics & visualization (`metrics.py`, `visualizer.py`)
- Q-Learning agent
- RL experiments (RL vs RL, FP vs RL)
- Results sections in report (Section 5)
- Presentation slides 6-11

**Estimated hours:** 40-50

---

### Together:
- Integration & testing
- Runner script (`runner.py`)
- Analysis & discussion (Report section 6)
- Report editing
- Presentation rehearsal

**Estimated hours:** 20-30

---

## 🔧 Technical Stack

### Core
- **Language:** Python 3.10+
- **Libraries:**
  - `numpy` - arrays, computations
  - `matplotlib` - plotting
  - `seaborn` (optional) - prettier plots

### Development
- **Version Control:** Git + GitHub
- **IDE:** VSCode / PyCharm
- **Report:** LaTeX (Overleaf) or Markdown

### Requirements File
```txt
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## ✅ Milestones & Checkpoints

### Checkpoint 1: Dec 27
- [ ] Git repo initialized
- [ ] Project structure created
- [ ] Theoretical notes complete
- [ ] `base_game.py` implemented

### Checkpoint 2: Jan 3
- [ ] Both games working
- [ ] Metrics functions implemented
- [ ] Basic plotting works

### Checkpoint 3: Jan 10
- [ ] FP agent complete
- [ ] RL agent complete
- [ ] Unit tests pass

### Checkpoint 4: Jan 17
- [ ] All experiments run
- [ ] Minimum 6 plots generated
- [ ] Results analyzed

### Checkpoint 5: Jan 24
- [ ] Draft report complete
- [ ] Preliminary slides ready

### Checkpoint 6: Jan 31 (1st Meeting)
- [ ] Live demo prepared
- [ ] 2-3 key plots polished
- [ ] Questions for professor ready

### Checkpoint 7: Feb 7
- [ ] Feedback implemented
- [ ] Final plots publication-ready
- [ ] Report finalized

### Checkpoint 8: Feb 14
- [ ] Presentation rehearsed
- [ ] Timing checked (15-20 min)
- [ ] Backup slides ready

---

## 🚨 Critical Success Factors

### 1. Simplicity > Complexity
- ✅ 2 games are enough
- ✅ 1 RL algorithm (Q-learning) suffices
- ❌ Don't add deep RL
- ❌ Don't add too many games

### 2. Plots = Points
- Spend 30% of time on visualization
- Make plots publication-ready
- Clear labels, legends, titles

### 3. Convergence Analysis
- Distance to Nash is THE key metric
- Show it prominently in report & presentation

### 4. Communication
- Sync every 2 days
- Use Git branches
- Code reviews before merging

### 5. Testing
- Test early, test often
- Don't leave debugging for the end
- Write simple unit tests

---

## 🎁 Bonus Features (if time permits)

### Optional Additions:
- [ ] Stochastic game (state-dependent)
- [ ] Policy gradient RL
- [ ] Multi-agent RL (independent learners)
- [ ] Interactive visualization (Plotly)
- [ ] Comparison with other algorithms (Regret Matching)

**⚠️ Only do these if:**
- Core experiments done ✅
- Report draft ready ✅
- Professor suggests it in 1st meeting

**Otherwise:** Polish what you have!

---

## 📚 Reading List

### Must Read:
1. **Shoham & Leyton-Brown** - Chapter 3 (Mixed Strategies), Chapter 7 (Learning)
2. **Robinson (1951)** - "An iterative method of solving a game"
3. **Watkins (1989)** - "Learning from Delayed Rewards" (Q-learning)

### Good to Read:
4. **Sutton & Barto** - Reinforcement Learning book (Chapters 6-7)
5. **Fudenberg & Levine (1998)** - The Theory of Learning in Games

### Reference Material:
- NumPy documentation
- Matplotlib gallery
- Nash equilibrium computation algorithms

---

## 💡 Pro Tips

### Coding:
1. **Start simple** - Get Matching Pennies working first
2. **Modular code** - Easy to extend later
3. **Comment as you go** - Future you will thank you
4. **Git commits** - Small, frequent, descriptive

### Experiments:
1. **Save everything** - Raw data + plots
2. **Random seeds** - For reproducibility
3. **Progress bars** - Use `tqdm` for long runs
4. **Automate** - Scripts for all experiments

### Report:
1. **Write as you go** - Don't leave it all for the end
2. **Version control** - Use Git for LaTeX too
3. **Cite properly** - BibTeX from Google Scholar
4. **Proofread** - Read it out loud

### Presentation:
1. **Practice timing** - 15-20 minutes
2. **Tell a story** - Not just data dump
3. **Anticipate questions** - Prepare extra slides
4. **Backup plan** - PDF + video demo

---

## 🆘 Troubleshooting

### Common Issues:

**"FP not converging"**
- Check best response computation
- Verify belief updates
- Try longer runs (20k iterations)

**"RL oscillating wildly"**
- Decrease learning rate
- Increase epsilon (more exploration)
- Try decaying epsilon

**"Code too slow"**
- Vectorize with NumPy
- Profile with `cProfile`
- Reduce unnecessary logging

**"Plots look ugly"**
- Use seaborn styles
- Increase figure size
- Export as PDF, not PNG

---

## 📧 Communication Protocol

### With Professor:
- **Before 1st meeting:** Email outline + 2-3 plots
- **During meeting:** Take notes, ask specific questions
- **After meeting:** Email summary of agreed changes

### Between Team:
- **Sync:** Every 2 days (30 min call)
- **Updates:** Daily message (what you did)
- **Blockers:** Immediate notification
- **Code review:** Before merging to main

### Tools:
- Git: Version control + issues
- Discord/Slack: Daily communication
- Google Drive: Shared notes
- Overleaf: Collaborative LaTeX

---

## 🎯 Definition of Done

### Code:
- [ ] All functions documented
- [ ] Basic tests pass
- [ ] No hardcoded paths
- [ ] README with usage instructions

### Experiments:
- [ ] All 3 matchups complete
- [ ] Minimum 8 plots generated
- [ ] Results reproducible (seeds)
- [ ] Data saved in results/

### Report:
- [ ] 12-15 pages
- [ ] All sections complete
- [ ] References cited properly
- [ ] Figures numbered and captioned
- [ ] Proofread (no typos)

### Presentation:
- [ ] 15-18 slides
- [ ] Timing: 15-20 minutes
- [ ] Rehearsed 2+ times
- [ ] PDF backup ready

---

## 🚀 Getting Started

### Day 1 (Tomorrow):
```bash
# Initialize project
git init fictitious-play-rl
cd fictitious-play-rl

# Create structure
mkdir -p games agents experiments analysis results/plots results/data report

# Create files
touch games/base_game.py
touch agents/base_agent.py
touch requirements.txt
touch README.md

# First commit
git add .
git commit -m "Initial project structure"

# Start coding
code games/base_game.py
```

### First Task:
**Person 1:** Implement `base_game.py` + `matching_pennies.py`  
**Person 2:** Implement `metrics.py` (distance_to_nash function)

**Deadline:** Tonight, sync tomorrow morning

---

## 📞 Questions for 1st Meeting

Prepare these for Jan 27-31:

1. **Scope:** "Are 2 games (Matching Pennies + RPS) sufficient, or should we add a stochastic game?"

2. **Metrics:** "Which metrics are most important: distance to Nash, exploitability, or regret?"

3. **RL Variants:** "Is Q-learning enough, or should we implement another RL algorithm?"

4. **Presentation:** "Should we focus more on theory or experimental results?"

5. **Report:** "What level of mathematical rigor do you expect?"

---

## 🏁 Final Checklist (Feb 14)

### Before Presentation:
- [ ] Laptop fully charged
- [ ] Presentation on USB drive
- [ ] PDF backup ready
- [ ] Code demo tested
- [ ] Internet connection checked (if needed)
- [ ] Dressed appropriately
- [ ] Water bottle

### During Presentation:
- [ ] Speak clearly and slowly
- [ ] Make eye contact
- [ ] Point at specific parts of plots
- [ ] Stay within time limit
- [ ] Handle questions calmly

### After Presentation:
- [ ] Submit final report
- [ ] Submit code (GitHub link)
- [ ] Thank professor
- [ ] Celebrate! 🎉

---

**Good luck! You've got this! 💪**

---

*Last updated: Dec 16, 2024*
