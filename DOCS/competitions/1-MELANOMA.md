# Description of Melanoma Competition

## Overview
This competition invites participants to develop a machine learning model that **aids in detecting the possibility of melanoma**. The goal is to create a model that can identify patterns in data that are associated with an increased likelihood of melanoma in visual recognition.

### Objective
The primary objective is to develop a model that can analyze photos taken by users of their skin lesions or areas of concern. 
The model should **assist users** by providing a risk assessment or likelihood score that helps them decide if they should seek further medical advice.
As a result, best model will be released in Skin Scan mobile app to run locally on the phone, and a website that will host it, free for anyone to use. 

## Evaluation Criteria
Models will be evaluated based on described **performance metrics** of the model.
The evaluation will be calculaded on following metrics with described weights.

### Performance Metrics

 The models will be assessed on the following metrics with the corresponding weights:

| **Metric**  | **Description**                                       | **Weight** |
|-------------|-------------------------------------------------------|------------|
| **F-beta**  | Prioritizes recall, with a high beta to emphasize it. $\beta = 2$ | 0.60       |
| **Accuracy**| Measures the overall correctness of predictions.      | 0.30       |
| **AUC**     | Evaluates the model's ability to distinguish classes. | 0.10       |

### Mathematical Formulas

1. **F-beta Score $F\_\beta\$**

   
   $$F_\beta = \left(1 + \beta^2\right) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\left(\beta^2 \cdot \text{Precision}\right) + \text{Recall}}$$
   

   Where:
   - **$\beta$** is the weight of recall in the combined score
   - in our case $\beta = 2$ for higher recall importance

2. **Accuracy**

   $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Number of Samples}}$$

3. **Area Under the Curve (AUC)**

   AUC is the area under the Receiver Operating Characteristic (ROC) curve. It is calculated using the trapezoidal rule:

   $$\text{AUC} = \int_0^1 \text{TPR} \, d(\text{FPR})$$

   Where:
   - **TPR** = True Positive Rate
   - **FPR** = False Positive Rate


## Model Inputs and Outputs

### Inputs
- **Input Format**: Multiple images in JPEG or PNG format.
- **Input Features**: During preprocessing, images are resized to 224x224 pixels. Images are converted to numpy arrays with a datatype of `np.float32`, normalized to the range [0, 1].

### Outputs
- **Output Format**: A numerical value between 0 and 1, represented as a `float`. This value indicates the likelihood or risk score of the area of concern warranting further investigation.

### Submission Requirements
- **Model Submission**: Models must be submitted in ONNX format. They should be capable of handling dynamic batch sizes and accept inputs with the shape `(batch , 3 , 224 , 224)`, where `batch` represents the batch dimension. This ensures that the model can process a variable number of images in a single batch.


## Rules and Guidelines

- **Timeline**:
 - every day competition will be run one or more times a day. Timings are defined in [competition_config.json](../../config/competition_config.json)
 - couple of minutes before start of competition, new part of dataset will be published for testing.
- Results of competition will be available on the dashboard


## Examplary code for model generation

This code demonstrates how to export a custom convolutional neural network (CNN) model from PyTorch to the ONNX format

```
import torch
import torch.nn as nn
import torch.onnx
import onnx


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Example model architecture
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Determine the correct size of the input features for the fully connected layer
        self.fc1_input_dim = 32 * 110 * 110  # Update this based on the output size from forward pass
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)  
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Load the model with the appropriate architecture
model = CustomCNN()
# model.load_state_dict(torch.load('path_to_your_model_weights.pth'))  # Load the model weights
model.load_state_dict(torch.load('path_to_model.pth'))  # Load the model weights
model.eval()  # Set the model to evaluation mode

# Prepare example input (e.g., a batch of images)
example_input = torch.randn(2, 3, 224, 224)  # Example input; adjust the shape based on your model's input size

# Export the model to ONNX format
onnx_path = 'path_to_export_model.onnx'
torch.onnx.export(
    model,
    example_input,
    onnx_path,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model exported to {onnx_path}")

# Verify the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")
```

## FAQ

### System Requirements and Setup

**Q: What are the machine requirements to run the miner?**  
A: You'll need **at least 8GB of RAM** to run the test dataset locally, along with all the necessary dependencies installed.

**Q: Do I need to create a new model/dataset on Hugging Face to run a miner?**  
A: Yes, each miner needs to **train their own model** and host it in their own public Hugging Face repository. You can submit it using the neurons/miner.py script (see [miner.md](../miner.md) for details).

---

### Submission Process

**Q: Do I need to submit the model and code at a specific time every day?**  
A: No, you only need to submit your model once, whenever you want. It will automatically be fetched multiple times per day specified in [competition_config.json](../../config/competition_config.json). If your model performs the best during a run, you'll receive the emission. The dataset is refreshed daily, so your model will be continuously tested against an ever-growing dataset.ur model performs best during a run, you'll receive the emission reward.

**Q: What is the submission deadline?**  
A: There’s **no deadline**. The competition runs indefinitely with new cancer-related competitions being introduced over time.

**Q: Is the submission time zone in UTC?**  
A: Yes, all competition runs are based on **UTC**. Once you submit your model, it will be tested automatically during each competition cycle.

**Q: Is there a limit of model size or inference time?**  
A: Currently, there is no strict limit on model size or inference time. However, since the end goal is to deploy the model on a mobile app, it's important to keep in mind that the size should be reasonable to ensure optimal performance and efficiency on mobile devices.

---

### Other

**Q: Where can I find the competition IDs and datasets?**  
A: Competition details, including **IDs and datasets**, can be found in the [competition_config.json](../../config/competition_config.json). Datasets are available on Hugging Face, and new datasets are released a few minutes before each competition starts.<br/>
https://huggingface.co/datasets/safescanai/melanoma-competition/tree/main

**Q: Can others access my Hugging Face repo?**  
A: Yes, since your repository is **public**, others can view and even use your model. However, a protected scoring mechanism ensures fairness in the competition.

---