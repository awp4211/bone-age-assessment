# Bone Age Assignment via Deep Learning Technique
In this paper, we use active learning and query by committee strategy to actively select training sample and segment hand from original X-Ray images.

We define the bone age assignment task a as regression task. With the segmented results, we fine-tune 3 state-of-the-art multi-scale CNN model, inception v3, inception resnet v2 and xception, to predict real bone age of each X-ray image.

Code Usage for AL segmentation
1. run al_segmentation/step1_corase_segment.py to generate corase segmentation with traditional image preprocessing method, including OTSU and Giant Component.
2. run al_segmentation/step2_interactive_segmentation.py to execute interactive program. After each training epoch, oracle would like to manually segment most uncertain samples and add them to training set.
   Before running next training epoch, oracle may like to use script step2_image_2_h5.py to add new annotated samples to training set. Note that in this interactive process, new data should be located at `RSNA_GT_NEW = "../data/gt/new"` with names of `xxx.png (ori)` and `xxx_bin.png(GT)`.
3. run al_segmentation/step3_check_data.py to make sure there are 12611 segmented hand images.
4. run al_segmentation/step3_gen_segmentation_result.py to use trained segment model to inference segmentation result.
5. run al_segmentation/step3_gen_segmentation_with_enhance.py to generate final segment results. The segmented hand with be located at `RSNA_SEG_ALL="../data/segment_all"` with the size of 512*512
6. run step4_regression_multi_gpu.py to fine-tune pretrained CNN model and make BAA.
7. run step4_regression_with_cam.py to train model and run step6_cam.py to generate CAM
8. run step5_statiscal_data.py to gather statistics of training result, including MAE, MSE and other metrics.

Code Usage for paced transfer learning.
1. run paced_transfer/step1_ptl_gen.py to train BAA model with PTL
2. run paced_transfer/step2_conventional_transfer.py to train BAA model without PTL
3. run paced_transfer/step5_cam_inference.py to generate CAM results
4. run paced_transfer/step6_visualize.py to generate training loss figures