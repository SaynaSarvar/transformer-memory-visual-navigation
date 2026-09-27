import torch

from models.observation_encoder import ObservationEncoder
from models.transformer_memory import TransformerMemory



class NavigationAgent:
    def __init__(
        self,
        sac_agent,
        embedding_dim=256,
        memory_length=8,
        device="cpu"
    ):
        self.device = torch.device(device)

        self.sac_agent = sac_agent

        self.observation_encoder = ObservationEncoder(
            visual_dim=128,
            goal_dim=128,
            embedding_dim=embedding_dim
        ).to(self.device)

        self.transformer_memory = TransformerMemory(
            embedding_dim=embedding_dim,
            num_heads=4,
            memory_length=memory_length
        ).to(self.device)

        self.memory = []
        self.memory_length = memory_length

    def preprocess_observation(self, observation):

        image = observation["image"]
        goal = observation["goal"]

        image = torch.tensor(
            image,
            dtype=torch.float32,
            device=self.device
        )

        goal = torch.tensor(
            goal,
            dtype=torch.float32,
            device=self.device
        )

        # HWC → CHW
        image = image.permute(2, 0, 1)

        # Add batch dimension
        image = image.unsqueeze(0)
        goal = goal.unsqueeze(0)

        # Normalize RGB image
        image = image / 255.0

        return image, goal    
    @torch.no_grad()
    def get_belief_state(self, observation):

        image, goal = self.preprocess_observation(
            observation
        )

        # Observation → embedding
        embedding = self.observation_encoder(
            image,
            goal
        )

        # Add embedding to temporal memory
        self.memory.append(
            embedding
        )

        # Keep only recent embeddings
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)

        # Convert list to tensor
        memory_tensor = torch.cat(
            self.memory,
            dim=0
        )

        # Add sequence dimension
        memory_tensor = memory_tensor.unsqueeze(0)

        # Transformer
        belief_state, attention_weights = (
            self.transformer_memory(
                embedding,
                memory_tensor
            )
        )

        return belief_state, attention_weights
    @torch.no_grad()
    def act(self, observation):
        belief_state, attention_weights = self.get_belief_state(
            observation
        )

        action, log_prob, _, _ = self.sac_agent.actor.sample(
            belief_state
        )

        action = action.squeeze(0).cpu().numpy()

        return action, belief_state, attention_weights

    @torch.no_grad()
    def act_from_state(self, belief_state):
        action, log_prob, _, _ = self.sac_agent.actor.sample(
            belief_state
        )

        action = action.squeeze(0).cpu().numpy()

        return action
    def reset_memory(self):
        self.memory = []