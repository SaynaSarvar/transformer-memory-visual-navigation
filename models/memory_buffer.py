import torch


class MemoryBuffer:

    def __init__(self, memory_length=8):

        self.memory_length = memory_length

        self.memory = []


    def reset(self):

        self.memory = []


    def add(self, embedding):

        # Remove batch dimension
        embedding = embedding.squeeze(0)

        self.memory.append(
            embedding.detach()
        )

        # Keep only the latest embeddings
        if len(self.memory) > self.memory_length:

            self.memory.pop(0)


    def get_memory(self):

        if len(self.memory) == 0:

            return None

        memory = torch.stack(
            self.memory,
            dim=0
        )

        # Add batch dimension
        memory = memory.unsqueeze(0)

        return memory


    def __len__(self):

        return len(self.memory)