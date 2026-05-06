# SpaPOT loss search round

This round is visual-first: metrics are recorded, but final choice must inspect the montage and per-time comparison images.

## Variants

- `r14_notype_nomass_state6040_spatial020`: Classifier-based labels remove the need for KNN-type matching during evaluation. Test whether dropping the cell-type prior gives a less scattered, more natural silhouette.
- `r14_typeprior_nomass_state7030_spatial020`: Keep a weak type prior but move closer to the original stVCR-like 70/30 state matching; external mass losses remain off.
- `r14_zt_nomass_state6040_spatial020`: Ablate U(s,z,t) versus U(z,t): keep the same spatial vector field and matching but make gene potential depend only on latent state and time.

## Metric summary

| variant | final_spatial_grid_iou | final_centroid | final_chamfer | mean_mass_ratio_abs_error | final_label_prop_l1 |
| --- | --- | --- | --- | --- | --- |
| r14_notype_nomass_state6040_spatial020 | 0.0295 | 0.1795 | 0.0293 | 0.2508 | 0.1554 |
| r14_typeprior_nomass_state7030_spatial020 | 0.0337 | 0.0863 | 0.0267 | 0.1716 | 0.1455 |
| r14_zt_nomass_state6040_spatial020 | 0.0346 | 0.6177 | 0.0309 | 0.0167 | 0.3901 |

## Current ranking

- Best final IoU: `r14_zt_nomass_state6040_spatial020`.
- Best final centroid: `r14_typeprior_nomass_state7030_spatial020`.
- Best mass ratio: `r14_zt_nomass_state6040_spatial020`.

## Montage

![final montage](/Users/wanghongye/python/stVCR/spapot/runs/spapot_loss_search/round14_classifier_nomass_hybrid/final_visual_montage.png)

## Visual inspection notes

- `r14_typeprior_nomass_state7030_spatial020` is the best visual choice in this round. The final silhouette is still not perfect, but it preserves the blue body and orange branch more coherently than the other two variants, and it avoids the severe centroid drift of `U(z,t)`.
- `r14_notype_nomass_state6040_spatial020` shows that removing the cell-type prior does not help. The trajectory becomes more stretched and the final centroid error increases, even though the label proportions remain correlated.
- `r14_zt_nomass_state6040_spatial020` is a metric trap: final IoU is the highest, but the all-frame plot shows strong type mixing and a large spatial shift. This argues against using IoU alone and against reverting the gene potential to `U(z,t)` here.
- The next useful direction is not stronger mass loss. It is a stVCR-like hybrid with weak type prior, 70/30 or 65/35 state matching, modest spatial OT, and possibly longer training or lower LR to improve silhouette without adding rollout/mass instability.
