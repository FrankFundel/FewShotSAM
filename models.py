import torch
from torch import nn
from segment_anything import sam_model_registry
from utils import SAMPostprocess

class SAM_Baseline(torch.nn.Module):
    def __init__(self, multi_output=True):
        super(SAM_Baseline, self).__init__()
        self.sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
        self.img_size = self.sam_model.image_encoder.img_size
        self.postprocess_masks = SAMPostprocess(self.img_size)
        self.multi_output = multi_output

    def forward(self, embeddings, points):
        labels = torch.ones(embeddings.shape[0], 1)
        labels.to(points.device)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
          points=(points.unsqueeze(1), labels),
          boxes=None,
          masks=None
        )
        masks, iou_predictions = self.sam_model.mask_decoder(
          image_embeddings=embeddings,
          image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
          sparse_prompt_embeddings=sparse_embeddings,
          dense_prompt_embeddings=dense_embeddings,
          multimask_output=self.multi_output,
        )
        masks = self.postprocess_masks(masks)
        #masks = normalize(threshold(masks, 0.0, 0)).to(device) # sigmoid is set to true in dice
        if self.multi_output:
            return masks, iou_predictions # (B, 3, 1024, 1024), (B, 3)
        else:
            return masks # B, 1, 256, 256
    
class SAM_Concat(torch.nn.Module):
    def __init__(self, example_dim, example_dtype, multi_output=True):
        super(SAM_Concat, self).__init__()
        self.sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
        self.img_size = self.sam_model.image_encoder.img_size
        self.postprocess_masks = SAMPostprocess(self.img_size)
        self.multi_output = multi_output
        self.example_dim = example_dim
        self.example_layer = nn.Linear(self.example_dim, 256, dtype=example_dtype) # B * K, N -> B * K, 256

    def forward(self, embeddings, points, example_tokens):
        labels = torch.ones(embeddings.shape[0], 1)
        labels.to(points.device)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
          points=(points.unsqueeze(1), labels),
          boxes=None,
          masks=None
        )
        
        b, k, n = example_tokens.shape
        example_tokens = self.example_layer(example_tokens.reshape(b * k, n)).reshape(b, k, 256)
        sparse_embeddings = torch.cat([sparse_embeddings, example_tokens], 1)
        
        masks, iou_predictions = self.sam_model.mask_decoder(
          image_embeddings=embeddings,
          image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
          sparse_prompt_embeddings=sparse_embeddings,
          dense_prompt_embeddings=dense_embeddings,
          multimask_output=self.multi_output,
        )
        masks = self.postprocess_masks(masks)
        #masks = normalize(threshold(masks, 0.0, 0)).to(device) # sigmoid is set to true in dice
        if self.multi_output:
            return masks, iou_predictions # (B, 3, 1024, 1024), (B, 3)
        else:
            return masks # B, 1, 256, 256

class SAM_GSA(torch.nn.Module):
    def __init__(self, example_dim, multi_output=True):
        super(SAM_GSA, self).__init__()
        self.sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
        print(self.sam_model.mask_decoder.transformer)
        self.sam_model.mask_decoder.transformer.gsa = True
        self.img_size = self.sam_model.image_encoder.img_size
        self.postprocess_masks = SAMPostprocess(self.img_size)
        self.multi_output = multi_output
        self.example_dim = example_dim
        self.example_layer = nn.Linear(self.example_dim, 256, dtype=torch.float16) # B * K, N -> B * K, 256

    def forward(self, embeddings, points, example_tokens):
        labels = torch.ones(embeddings.shape[0], 1)
        labels.to(points.device)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
          points=(points.unsqueeze(1), labels),
          boxes=None,
          masks=None
        )
        
        b, k, n = example_tokens.shape
        example_tokens = self.example_layer(example_tokens.reshape(b * k, n)).reshape(b, k, 256)
        
        masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multi_output,
            examples=example_tokens
        )
        masks = self.postprocess_masks(masks)
        #masks = normalize(threshold(masks, 0.0, 0)).to(device) # sigmoid is set to true in dice
        if self.multi_output:
            return masks, iou_predictions # (B, 3, 1024, 1024), (B, 3)
        else:
            return masks # B, 1, 256, 256