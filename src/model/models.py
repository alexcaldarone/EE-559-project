import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class MultiModalClassifier(nn.Module):
    def __init__(
            self, 
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            num_classes: int,
            transformer_hidden_dim: int,
            dropout_rate: float = 0.1,
            activation_function: str = "gelu"

    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.image_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.text_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.randn(1, 512, embed_dim))  # max_len = 512 tokens

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=transformer_hidden_dim,
            dropout=dropout_rate,
            activation=activation_function,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier head - this can bechanged quite a bit
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, image_embeds, text_embeds):
        """
        image_embeds: Tensor of shape [batch_size, num_img_tokens, embed_dim]
        text_embeds:  Tensor of shape [batch_size, num_text_tokens, embed_dim]
        """

        B, I, D = image_embeds.shape
        T = text_embeds.shape[1]

        # Expand special tokens to batch size
        cls_tok = self.cls_token.expand(B, -1, -1)
        img_tok = self.image_token.expand(B, 1, -1)
        txt_tok = self.text_token.expand(B, 1, -1)

        # Concatenate the sequence
        x = torch.cat([cls_tok, img_tok, image_embeds, txt_tok, text_embeds], dim=1)  # shape: [B, 1+1+I+1+T, D]

        # Add positional embeddings (truncate if needed) ?
        x = x + self.pos_embed[:, :x.size(1), :]

        # Pass through transformer
        x = self.transformer(x)

        # Take [CLS] token output
        cls_out = x[:, 0, :]  # [B, D]

        # Final classifier
        logits = self.classifier(cls_out)  # [B, num_classes]
        return logits