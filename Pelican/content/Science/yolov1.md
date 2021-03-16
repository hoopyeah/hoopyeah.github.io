Title: Notes on YOLOv1 Paper
Date: 2021-02-23 17:40
Tags: Neural Networks, AI, ML
Author: Cody Fernandez
Summary: Collecting the notes I took while reading the YOLOv1 paper

# YOLOv1
- Extremely fast, reasons globally about entire image, highly generalizable, fails on small objects
- Reframe object detection as a single regression problem.
- From image pixels to bounding box coordinates and class probabilities
- Divide an image into an $SxS$ grid
    - If the center of an object falls into that grid cell, that grid cell is responsible for detecting that object
    - A grid cell predicts $B$ bounding boxes and confidence scores
          - Confidence score covers likelihood box contains object and accuracy of box prediction
          - Confidence is $Pr(Objects)*IOU^{truth}_{pred}$
          - Intersection over union between predicted box and ground truth
    - A bounding box has 5 predictions: $x, y, w, h,$ and $confidence$.
        - $(x, y)$ is the center of the box relative to bounds of grid cell
        - $(w,h)$ relative to whole image
        - $confidence$ is IOU
    - A grid cell predicts $C$ conditional class probablilities $Pr(Class_i|Objects)$. These are conditiones on the grid cell containing the object.
        - Only predict one set of class probabilities per grid cell
    - At test time, multiply conditional class probabilities and individual box confidence predictions to get class-specific confidence scores for each box.
    - Evaluation: $S=7, B=2, C=20$. Final preditions is $7x7x30$ tensor.
- 24 convolutional layers, 2 fully connected layers *(dense?)*
- Use $1x1$ reduction layer and $3x3$ convolution layers
- Adding both convolutional and connected laters to pretrained networks can improve performance
- Normalize bounding box $(w, h)$ by image $(w ,h)$ so it's in $(0,1)$
- Parametrize bounding box $(x, y)$ to be offsets of grid cell location, so it's in $(0,1)$
- Use linear activation on final layer and leaky ReLU everywhere else:
$$
    \phi(x) = \begin{cases} x, & \mbox{if } x>0 \\ 0.1x, & \mbox{otherwise} \end{cases}
$$

- Optimize sum-squared error in output
    - Easy (good)
    - Does not maximize average precision (bad)
    - Weights localization error equally with classification error (bad)
    - Model unstable due to overpowering gradient from cells containing objects. Object-lacking cells plunge to zero (bad)
        - Add $\lambda_{coord} = 5$ and $\lambda_{noobj}=0.5$. Increase loss of bounding box coordinate predictions and decrease loss of confidence predictions of object-lacking boxes (good)
    - Use square root of bounding box $(w,h)$ to properly weight small deviations in large boxes (good)
- Assign one predictor to be "responsible" for object prediction absed on hights current IOU with gound truth. This leads to bounding box predictor specialization, improving overall recall.
- The loss function only penalizes classification error if an object is present in that grid cell. It only penalizaes bounding box coordinate error if that predictor is "responsible" for that ground truth box.
- Grid design enforces spatial diversity in the bounding box predictions.
- YOLO imposes strong spatial constraints on bounding box predictions. It therefor struggles with small objects in groups.
- Struggles to generalize to new aspect ratios or configurations
- Uses coarse features for bounding box prediction.
- Main source of error is incorrect localizations. 
    