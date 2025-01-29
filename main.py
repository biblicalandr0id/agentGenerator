#main.py
import os
import sys
from pathlib import Path
import torch
import numpy as np
import json

# Get the absolute path of the directory containing main.py
main_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Create the path of neural_networks.py
neural_networks_path = main_dir / "neural_networks.py"
# Create the path of executor.py
executor_path = main_dir / "executor.py"
# Create the path of diagnostics.py
diagnostics_path = main_dir / "diagnostics.py"
# Create the path of environment.py
environment_path = main_dir / "environment.py"
# Create the path of agent.py
agent_path = main_dir / "agent.py"
# Create the path of genetics.py
genetics_path = main_dir / "genetics.py"
# Create the path of embryo_namer.py
embryo_namer_path = main_dir / "embryo_namer.py"
# Create the path of embryo_generator.py
embryo_generator_path = main_dir / "embryo_generator.py"

# Add the directory containing genetics.py to the path. This lets us import genetics even from the same directory
sys.path.append(str(main_dir))

# Import from the files
from .neural_networks import NeuralAdaptiveNetwork
from .executor import AdaptiveExecutor
from .diagnostics import NeuralDiagnostics
from .environment import  EnhancedAdaptiveEnvironment, ResourceType, Resource
from .agent import AdaptiveAgent
from .genetics import create_genetic_core, GeneticCore
from .embryo_namer import EmbryoNamer
from .embryo_generator import EmbryoGenerator



if __name__ == "__main__":
    genetic_core = create_genetic_core(seed=42)
    embryo_namer = EmbryoNamer()
    generator = EmbryoGenerator(genetic_core, embryo_namer)
    embryo_file = generator.generate_embryo_file()
    size = (100, 100)
    complexity = 1.0
    environment = EnhancedAdaptiveEnvironment(size, complexity)

    input_size = 7
    hidden_sizes = [64, 64]
    output_size = hidden_sizes[0]
    neural_net = NeuralAdaptiveNetwork(input_size, hidden_sizes[0], output_size)

    diagnostics = NeuralDiagnostics(neural_net)

    agent_position = (50, 50)
    agent = AdaptiveAgent(genetic_core, neural_net, agent_position)
    environment.agents.append(agent)
    executor = AdaptiveExecutor(neural_net)

    env_state = environment.current_state
    env_state.resources = [
        Resource(type=ResourceType.ENERGY, quantity=100, position=(20, 20), complexity=0.2),
        Resource(type=ResourceType.MATERIALS, quantity=50, position=(70, 70), complexity=0.8)
    ]
    env_state.threats = [(80, 80)]
    
    agent.brain_interface.mainloop() # Start the brain UI

    neural_net.eval() # Set network to evaluation mode - BEFORE the loop

    for step in range(2):
        env_state = environment.current_state
        action, params = agent.decide_action(env_state)
        result = agent.execute_action(action, params, env_state)
        executor.record_validation_loss(result.reward)
        loss, outputs, importance = executor.execute(
            inputs=torch.tensor(agent.perceive_environment(env_state)).float().reshape(1, -1),
            targets=torch.tensor(np.zeros(hidden_sizes[0])).float().reshape(1, -1),
            context=torch.tensor([[0.0]]),
            diagnostics = diagnostics
        )
        diagnostics = agent.neural_diagnostics.monitor_network_health(
            inputs = torch.tensor(agent.perceive_environment(env_state)).float().detach().numpy().reshape(1,-1),
            targets = torch.tensor(np.zeros(hidden_sizes[0])).float().reshape(1,-1),
            context=torch.tensor([[0.0]]),
            epoch=env_state.time_step
        )

        print(json.dumps(diagnostics, indent=2))
        print(f"Step: {step+1}, Action: {action}, Reward: {result.reward:.4f}, Energy: {agent.energy:.2f}")
        environment.step([agent]) # Step the environment forward, passing in list of agents
        agent.mind.grow()
    print("\nAgent Status after 2 steps:", agent.get_status())