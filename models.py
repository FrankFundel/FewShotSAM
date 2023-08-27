import torch
from segment_anything import sam_model_registry
from utils import SAMPostprocess

class SAM_Baseline(torch.nn.Module):
    def __init__(self):
        super(SAM_Baseline, self).__init__()
        self.sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
        self.img_size = self.sam_model.image_encoder.img_size
        self.postprocess_masks = SAMPostprocess(self.img_size)

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
          multimask_output=True,
        )
        masks = self.postprocess_masks(masks)
        return masks, iou_predictions # (B, 3, 1024, 1024), (B, 3)

class SAM_Fine_Tune(torch.nn.Module):
    def __init__(self):
        super(SAM_Fine_Tune, self).__init__()
        self.sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
        self.img_size = self.sam_model.image_encoder.img_size
        self.postprocess_masks = SAMPostprocess(self.img_size)

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
          multimask_output=False, # fine-tuning such that the first decoder output is the best
        )
        masks = self.postprocess_masks(masks)
        #masks = normalize(threshold(masks, 0.0, 0)).to(device) # sigmoid is set to true in dice
        return masks # B, 1, 256, 256