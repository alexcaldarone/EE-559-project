import torch
import torch.nn as nn

# Define a simple binary classifier
class BinaryClassifier(nn.Module):
    def __init__(self, embedding_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_size * 2, 128)  # Combine image + text embeddings
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image_emb, text_emb):
        # Concatenate the image and text embeddings
        combined_emb = torch.cat((image_emb, text_emb), dim=1)
        x = torch.relu(self.fc1(combined_emb))
        x = self.fc2(x)
        return self.sigmoid(x)
