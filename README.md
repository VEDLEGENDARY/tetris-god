# Tetris AI - Deep Q-Learning Neural Network

A sophisticated Tetris AI powered by Deep Q-Learning (DQN) that learns to play Tetris through reinforcement learning. The agent uses a deep neural network to evaluate board states and optimize piece placement, maximizing score while minimizing board instability. Features a modern glassmorphic UI with real-time visualization and interactive controls.

## Features

### AI Architecture
- **Deep Q-Learning Agent (DQN)** with experience replay buffer and epsilon-greedy exploration
- **State Representation**: 4-dimensional vector analyzing lines cleared, holes, bumpiness, and cumulative height
- **Reward System**: +1 per piece placed, +(lines_cleared² × 10) for line clears, -2 for game over

### Training System
- **Automatic Model Checkpointing**: Saves `.keras` models every 50 episodes plus episode 1
- **Resumable Training**: Continues from the last saved episode with persistent state tracking
- **Independent Threading**: Training and visualization run concurrently without interference

### User Interface
- **Real-Time Visualization**: Watch the AI play with adjustable speed (0.5x-10x)
- **Performance Charts**: Track scores and learning progress over time

### Technical Optimizations
- **CPU/GPU Acceleration**: Automatic GPU detection with oneDNN optimizations
- **Mixed Precision (FP16)**: Faster training on compatible hardware

## Usage

1. **Start Training**: Click "Start Training" to begin AI learning (models save automatically)
2. **Visualize**: Select an episode and click "Start Visualization" to watch the AI play
3. **Adjust Parameters**: Modify training hyperparameters in the control panel
4. **Reset**: Clear all training data to start fresh

## Model Files

- **models/episode_X.keras**: Checkpoints every 50 episodes
- **models/_training_state.json**: Persistent training state
- **best.keras**: Best-performing model

## Performance

The AI typically achieves:
- Episode 1+: ~20-50 pieces placed
- Episode 100+: ~100-300 pieces placed  
- Episode 200+: ~300+ pieces with multi-line clears

Training time: 1-2 minutes for 1-1000 episodes
               ~5 minutes for 1000-2000 episodes
               ~10/♾️ based on parameters.

Please set a reasonably low max score limit after 1800 episodes since at about 2000/2200 episodes, the model may hit exponentially large scores and will take 10 minutes - multiple hours for each 10-20 episodes onwards.

## Building Executable

To create a standalone `.exe` file:
```bash
python -m PyInstaller tetris_gui.spec --clean --noconfirm
```