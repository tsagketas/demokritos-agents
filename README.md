# Fictitious Play & Reinforcement Learning

Learning equilibria in repeated zero-sum games.

## 🐳 Docker Setup

### Prerequisites
- Docker installed
- Docker Compose installed

### Quick Start

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Run experiments:**
   ```bash
   docker-compose up
   ```

3. **Run specific experiment:**
   ```bash
   docker-compose run --rm game-learning python -m experiments.fp_vs_fp
   ```

4. **Run interactive shell:**
   ```bash
   docker-compose run --rm game-learning bash
   ```

5. **Generate Hunter–Prey GIFs** (saves to `results/plots/hunter_prey_early.gif`, `hunter_prey_best.gif`):
   ```bash
   docker-compose run --rm game-learning python -m analysis.gif_maker
   ```

### Development Mode

The `docker-compose.yml` is configured with volume mounts, so your code changes are reflected immediately:

```bash
# Start container in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Building Docker Image Only

```bash
docker build -t fictitious-play-rl .
```

### Running Without Docker Compose

```bash
docker run -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/results:/app/results \
  fictitious-play-rl
```

## 📁 Project Structure

```
project/
├── games/              # Game implementations
├── agents/             # FP and RL agents
├── experiments/        # Experiment runners
├── analysis/           # Metrics and visualization
├── results/            # Output plots and data
└── report/             # LaTeX/Markdown report
```

## 🔧 Requirements

See `requirements.txt` for Python dependencies.

## 📚 Documentation

- **`final_plan.md`** - Complete project documentation and plan
- **`EXPERIMENTS.md`** - Guide for running experiments
- **`PROGRESS.md`** - Current progress and what's left to do

## 🧪 Running Experiments

See `EXPERIMENTS.md` for detailed instructions.

**Quick Start:**
```bash
# FP vs FP
docker-compose run --rm game-learning python -m experiments.fp_vs_fp

# RL vs RL
docker-compose run --rm game-learning python -m experiments.rl_vs_rl

# FP vs RL
docker-compose run --rm game-learning python -m experiments.fp_vs_rl

# Hunter–Prey GIFs (animations in results/plots/)
docker-compose run --rm game-learning python -m analysis.gif_maker
```

6. **Παρουσίαση (HTML → PDF):**
   ```bash
   # 1. Compose slides → project_presentation.html (στο game-learning/)
   docker-compose run --rm game-learning python presentation/compose_slides.py -d presentation/game-learning

   # 2. HTML → PDF (μετά από docker-compose build για playwright/img2pdf)
   docker-compose run --rm game-learning python presentation/html_to_pdf.py
   ```

