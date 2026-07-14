开始训练（YOLO），训练集共 1296 张图片
Creating new Ultralytics Settings v0.0.6 file ✅
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see <https://docs.ultralytics.com/quickstart/#ultralytics-settings>.
Downloading <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt> to 'yolov8s.pt': 100% ━━━━━━━━━━━━ 21.5MB 21.5MB/s 1.0s
Ultralytics 8.4.95 🚀 Python-3.10.11 torch-2.6.0+cu124 CUDA:0 (Tesla V100-SXM2-32GB, 32501MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=True, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/root/data/dataset.yaml, degrees=15.0, deterministic=True, device=0, dfl=1.5, dis=6.0, distill_model=None, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8s.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=train, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/root/runs/detect, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/root/runs/detect/train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading <https://ultralytics.com/assets/Arial.ttf> to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 134.2MB/s 0.0s
Overriding model.yaml nc=80 with nc=8

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]
  3                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  5                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]
  8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    591360  ultralytics.nn.modules.block.C2f             [768, 256, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 16                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 19                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]
 22        [15, 18, 21]  1   2119144  ultralytics.nn.modules.head.Detect           [8, 16, None, [128, 256, 512]]
Model summary: 130 layers, 11,138,696 parameters, 11,138,680 gradients, 28.7 GFLOPs

Transferred 349/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt> to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 259.4MB/s 0.0s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2143.4±888.2 MB/s, size: 388.0 KB)
train: Scanning /root/data/dataset/labels/train... 648 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 648/648 924.4it/s 0.7s
train: New cache created: /root/data/dataset/labels/train.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1668.1±788.6 MB/s, size: 93.5 KB)
val: Scanning /root/data/dataset/labels/val... 121 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 121/121 814.9it/s 0.1s
val: New cache created: /root/data/dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.000833, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Plotting labels to /root/runs/detect/train/labels.jpg...
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to /root/runs/detect/train
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      6.96G      2.699      6.661      2.526         57        640: 100% ━━━━━━━━━━━━ 21/21 1.6it/s 13.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.8s/it 5.5s
                   all        121        340      0.125      0.337      0.192     0.0706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      8.71G      2.114      2.356      1.962         60        640: 100% ━━━━━━━━━━━━ 21/21 3.9it/s 5.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.3s
                   all        121        340      0.293      0.426      0.313     0.0947

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      8.76G      2.021      1.938      1.881         59        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.5it/s 0.8s
                   all        121        340       0.56      0.407      0.352      0.126

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100       8.8G      1.986       1.79      1.856         53        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.4it/s 0.6s
                   all        121        340      0.545       0.41      0.386      0.147

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      8.84G      1.968      1.787      1.858         53        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.0s
                   all        121        340      0.427      0.402      0.365      0.124

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      8.88G      1.934      1.716      1.807         51        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.7it/s 0.7s
                   all        121        340       0.43      0.454      0.389      0.138

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      8.91G      1.935      1.681      1.824         35        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.5it/s 0.8s
                   all        121        340      0.516      0.451      0.362      0.131

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      8.96G      1.899      1.653      1.805         42        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.0s
                   all        121        340      0.432      0.455      0.425       0.17

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      8.99G      1.917      1.607      1.807         48        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.1s
                   all        121        340      0.472      0.405      0.345      0.142

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      9.03G      1.867      1.577      1.787         38        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.0it/s 1.0s
                   all        121        340      0.425      0.496      0.426      0.194

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      9.06G      1.839      1.558      1.745         36        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 3.0it/s 0.7s
                   all        121        340      0.634       0.48      0.445      0.156

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      9.11G      1.829      1.506      1.756         43        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.4it/s 0.8s
                   all        121        340       0.63      0.454      0.451      0.157

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      9.14G      1.821      1.496      1.748         51        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.1s
                   all        121        340      0.537      0.487      0.483      0.196

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      9.19G      1.789      1.473      1.719         40        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.8it/s 1.1s
                   all        121        340      0.635      0.499      0.525      0.224

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      9.22G      1.794      1.438       1.74         62        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.0it/s 1.0s
                   all        121        340       0.54      0.535      0.473      0.179

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      9.26G      1.805      1.451      1.737         57        640: 100% ━━━━━━━━━━━━ 21/21 3.8it/s 5.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.0it/s 1.0s
                   all        121        340      0.564      0.544      0.523      0.206

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      9.29G      1.783      1.428      1.726         43        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9s/it 3.9s
                   all        121        340      0.551      0.519      0.523      0.235

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      9.34G       1.74      1.379      1.693         56        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all        121        340      0.641      0.526      0.538      0.228

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      9.37G      1.732      1.361      1.674         65        640: 100% ━━━━━━━━━━━━ 21/21 4.6it/s 4.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all        121        340      0.735      0.522      0.539      0.207

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      9.42G      1.744      1.348      1.669         42        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.539      0.515      0.499      0.184

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      9.45G      1.696      1.335      1.647         50        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.2s
                   all        121        340      0.556      0.502      0.515      0.217

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100      9.49G      1.717      1.337      1.661         62        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.7s
                   all        121        340      0.627      0.549      0.551      0.238

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      9.53G      1.709      1.338      1.647         55        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.612      0.546      0.521      0.207

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      9.57G      1.681      1.305      1.639         49        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1it/s 1.8s
                   all        121        340      0.538      0.564      0.546      0.229

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100       9.6G      1.654      1.282      1.629         50        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.0s
                   all        121        340      0.624      0.564      0.567      0.255

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      9.65G      1.642      1.265      1.626         49        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all        121        340      0.756      0.533      0.531      0.224

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      9.68G      1.643      1.291      1.639         46        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.3s
                   all        121        340      0.595      0.507      0.515      0.239

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      9.72G      1.656      1.269      1.626         42        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.7s
                   all        121        340      0.627      0.541      0.554      0.256

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      9.76G      1.623      1.262      1.608         52        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.7s
                   all        121        340      0.657      0.528      0.563      0.272

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100       9.8G      1.601      1.258      1.615         54        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.648       0.55      0.552      0.252

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100      9.83G      1.646      1.257      1.619         46        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.6s/it 3.1s
                   all        121        340      0.544      0.553      0.543      0.241

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      9.88G      1.614       1.25       1.59         51        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.2s
                   all        121        340      0.635      0.557       0.57      0.241

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      9.91G      1.594      1.235      1.586         29        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.688      0.533       0.57      0.268

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100      9.96G      1.615      1.224      1.591         53        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.7s
                   all        121        340      0.596      0.567      0.556       0.25

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      9.99G      1.623      1.242       1.62         49        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.0s
                   all        121        340       0.58      0.551       0.56      0.235

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100        10G      1.563      1.185      1.583         47        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.7s
                   all        121        340      0.655      0.557      0.574      0.258

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      10.1G      1.581       1.19      1.573         37        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.0s/it 2.1s
                   all        121        340      0.633      0.545      0.566      0.236

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      10.1G       1.57      1.165      1.566         56        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all        121        340      0.652      0.568      0.585      0.259

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      10.1G      1.566      1.196      1.572         37        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.7s
                   all        121        340      0.603      0.599      0.572      0.251

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      10.2G      1.562      1.167      1.561         45        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.0s/it 2.1s
                   all        121        340      0.633      0.578      0.582      0.237

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      10.2G      1.544      1.162      1.559         44        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.656      0.574      0.585      0.261

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100      10.3G      1.583      1.183      1.568         57        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.579      0.604      0.569       0.26

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      10.3G      1.545      1.158      1.562         38        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.3s
                   all        121        340      0.615      0.554      0.562      0.253

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      10.3G      1.563      1.164       1.55         48        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.616      0.569      0.564      0.272

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      10.4G      1.534      1.141      1.538         46        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.2s
                   all        121        340      0.613      0.541      0.559      0.274

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      10.4G      1.536      1.161      1.534         63        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.0it/s 1.9s
                   all        121        340      0.646       0.58      0.562      0.256

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      10.5G      1.505      1.103        1.5         70        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.664      0.593      0.583      0.255

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100      10.5G      1.513      1.128      1.544         46        640: 100% ━━━━━━━━━━━━ 21/21 4.7it/s 4.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9s/it 3.9s
                   all        121        340       0.59      0.603      0.579      0.254

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      10.5G      1.493      1.103      1.518         52        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1it/s 1.9s
                   all        121        340      0.701      0.539      0.568      0.269

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      10.6G      1.488      1.103      1.514         51        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.2s
                   all        121        340      0.688      0.571      0.585      0.272

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      10.7G      1.491      1.118      1.519         49        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.0s/it 2.0s
                   all        121        340       0.59      0.566      0.562      0.259

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      10.8G      1.466      1.094      1.508         34        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all        121        340      0.608      0.587      0.577      0.256

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100      10.8G      1.476      1.088      1.507         52        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 2.9s
                   all        121        340      0.657      0.584      0.584      0.264

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      10.9G       1.47      1.079      1.492         48        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all        121        340      0.655      0.564      0.573      0.276

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      11.1G      1.479      1.097        1.5         41        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.1s
                   all        121        340      0.708      0.577      0.586      0.275

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      11.1G      1.451      1.052      1.482         50        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.5s
                   all        121        340      0.719      0.577      0.597      0.267

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      11.1G      1.447      1.062      1.478         61        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.707      0.596      0.609      0.277

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      11.4G      1.434      1.035      1.475         45        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.3s
                   all        121        340      0.652      0.577       0.57      0.263

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100      11.4G      1.438      1.031      1.476         30        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.5s
                   all        121        340      0.651       0.59      0.579      0.264

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      11.4G      1.425      1.056      1.479         48        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.3s
                   all        121        340      0.604      0.581       0.56      0.258

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      11.6G      1.442      1.042       1.49         56        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.6s/it 3.1s
                   all        121        340       0.66      0.592      0.576      0.263

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100      11.6G      1.443      1.058      1.464         47        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.0s
                   all        121        340      0.647       0.59      0.587      0.258

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      11.7G      1.422      1.042      1.458         48        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all        121        340       0.67      0.596      0.593      0.261

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      11.8G      1.417      1.038      1.471         49        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.702      0.557      0.575      0.258

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      11.9G      1.398      1.029      1.456         49        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.641      0.599      0.587      0.262

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100        12G      1.418      1.019      1.461         58        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all        121        340       0.72      0.599      0.605      0.285

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      12.1G      1.407      1.013       1.45         55        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.1s
                   all        121        340      0.625      0.603      0.595      0.281

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      12.2G      1.368      0.998      1.441         58        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.5s/it 3.0s
                   all        121        340       0.62      0.613      0.585      0.277

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      12.2G      1.377     0.9985      1.442         36        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.1s
                   all        121        340      0.681      0.592      0.576      0.271

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100      12.4G      1.375      1.003      1.438         53        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.678       0.57      0.587      0.262

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100      12.4G      1.372     0.9935      1.446         42        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.7s/it 3.3s
                   all        121        340      0.656      0.595      0.592       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100      12.6G      1.351     0.9803      1.418         64        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.643      0.579      0.571      0.257

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100      12.6G      1.392      0.992       1.43         64        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.6s/it 3.2s
                   all        121        340      0.644      0.587      0.578      0.276

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100      12.7G      1.319     0.9525      1.398         52        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.0s/it 2.0s
                   all        121        340      0.657      0.579      0.581      0.277

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100      12.7G      1.356     0.9761      1.431         36        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.8s/it 3.6s
                   all        121        340       0.67      0.591      0.582      0.282

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100      12.9G      1.339     0.9704      1.408         54        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.3s
                   all        121        340      0.637      0.596      0.577      0.265

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100      12.9G      1.328     0.9578      1.405         41        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.8s/it 3.5s
                   all        121        340      0.685      0.574      0.589      0.274

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100        13G      1.321     0.9617      1.406         49        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.5s
                   all        121        340      0.683      0.574      0.598      0.275

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100      13.1G       1.34     0.9684       1.41         46        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.5s
                   all        121        340      0.696      0.585      0.591       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100      13.2G      1.342     0.9607      1.409         67        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.8s/it 3.6s
                   all        121        340      0.698      0.585      0.587       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100      13.3G      1.304     0.9459      1.408         43        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.682      0.583      0.591       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100      13.4G      1.295     0.9375      1.392         58        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.5s
                   all        121        340      0.667      0.575      0.586      0.278

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100      13.4G      1.281     0.9391      1.395         38        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.658      0.581      0.584      0.267

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100      13.5G      1.262     0.9048      1.375         48        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.677      0.578      0.604      0.274

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100      13.7G      1.281     0.9133      1.392         34        640: 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.0s/it 2.1s
                   all        121        340      0.639      0.571      0.581      0.286

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100      13.7G      1.288     0.9109      1.385         56        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9s/it 3.8s
                   all        121        340      0.652      0.606      0.597      0.284

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100      13.8G      1.275     0.9103      1.381         53        640: 100% ━━━━━━━━━━━━ 21/21 4.6it/s 4.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.7s
                   all        121        340      0.642      0.615      0.592      0.278

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100      13.9G      1.276     0.9132      1.365         60        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.637      0.615      0.583      0.272

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100      13.9G      1.268     0.9114       1.37         58        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.3s
                   all        121        340      0.672      0.607      0.597      0.279

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100      14.1G      1.236     0.8914      1.357         39        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.6s
                   all        121        340      0.644      0.604      0.582      0.281
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100      14.3G      1.303     0.8811      1.446         17        640: 100% ━━━━━━━━━━━━ 21/21 1.6it/s 12.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.8it/s 0.7s
                   all        121        340       0.68       0.58      0.581      0.276

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100      14.4G      1.242     0.8279      1.427         21        640: 100% ━━━━━━━━━━━━ 21/21 3.2it/s 6.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.0it/s 1.0s
                   all        121        340      0.683      0.561      0.578      0.277

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100      14.4G      1.225     0.8147      1.414         21        640: 100% ━━━━━━━━━━━━ 21/21 3.6it/s 5.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.0s
                   all        121        340      0.654      0.586      0.573      0.271

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100      14.4G      1.184     0.7903      1.405         25        640: 100% ━━━━━━━━━━━━ 21/21 3.9it/s 5.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.6it/s 0.8s
                   all        121        340       0.67      0.573      0.582      0.277

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100      14.5G      1.193      0.795      1.387         19        640: 100% ━━━━━━━━━━━━ 21/21 3.9it/s 5.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.9it/s 1.0s
                   all        121        340      0.697      0.567      0.584       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100      14.6G      1.172     0.7731      1.368         26        640: 100% ━━━━━━━━━━━━ 21/21 4.2it/s 5.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.6it/s 1.2s
                   all        121        340      0.677      0.557      0.574      0.276

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100      14.6G      1.178     0.7892      1.374         18        640: 100% ━━━━━━━━━━━━ 21/21 4.0it/s 5.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.2s/it 2.4s
                   all        121        340      0.679      0.547       0.57      0.272

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100      14.8G      1.177     0.7841      1.375         20        640: 100% ━━━━━━━━━━━━ 21/21 4.4it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4s/it 2.8s
                   all        121        340      0.706      0.559      0.578      0.281

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      14.8G      1.168     0.7774      1.371         25        640: 100% ━━━━━━━━━━━━ 21/21 4.5it/s 4.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.3s/it 2.5s
                   all        121        340      0.705      0.564      0.582      0.283

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      14.9G      1.174     0.7682       1.37         19        640: 100% ━━━━━━━━━━━━ 21/21 4.3it/s 4.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.1s/it 2.3s
                   all        121        340      0.704      0.565      0.582      0.283

100 epochs completed in 0.233 hours.
Optimizer stripped from /root/runs/detect/train/weights/last.pt, 22.5MB
Optimizer stripped from /root/runs/detect/train/weights/best.pt, 22.5MB

Validating /root/runs/detect/train/weights/best.pt...
Ultralytics 8.4.95 🚀 Python-3.10.11 torch-2.6.0+cu124 CUDA:0 (Tesla V100-SXM2-32GB, 32501MiB)
Model summary (fused): 73 layers, 11,128,680 parameters, 0 gradients, 28.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 2.4s/it 4.9s
                   all        121        340      0.613      0.579      0.577      0.291
                  cart         20         21      0.872      0.952      0.977       0.64
                 track        111        115      0.759      0.783      0.814      0.437
                string         53         55      0.469      0.218       0.26      0.111
           dynamometer         51         52      0.499      0.538      0.493      0.192
          wooden_block         52         52      0.762      0.827        0.8      0.338
            iron_block         36         45      0.316      0.154      0.121     0.0283
Speed: 0.1ms preprocess, 4.3ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to /root/runs/detect/train

训练完成，最优权重: /root/runs/detect/train/weights/best.pt
mAP@0.5: 0.5774
