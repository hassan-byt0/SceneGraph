# Scene Graph Reasoning with Object Detection and Graph Neural Networks

This repository demonstrates how to integrate object detection with scene graph reasoning using PyTorch, Detectron2, and PyTorch Geometric. The pipeline includes detecting objects in an image, constructing a scene graph, and applying a Graph Neural Network (GNN) for reasoning over the scene. Additionally, it incorporates rule-based reasoning using a predefined knowledge base.

### Input
![Input](https://drive.google.com/uc?id=1WKdbn8xUjZUMceW39HgdPmOVdYsTTctI "Input")
### Test Output
![Test Output](https://drive.google.com/uc?id=15ZnFYf85NEHp5Xnj6fXMkkXwPRuUI-NL "Test Output")
Source on image: https://www.dreamstime.com/photos-images/driving.html 

for Reasoning refer Jupyter Notebook attached


## Features
1. **Object Detection**: Utilize Detectron2's pre-trained Faster R-CNN model for object detection.
2. **Scene Graph Creation**: Encode detected objects as nodes and their relationships as edges.
3. **Graph Neural Network**: Implement a GCN for scene graph reasoning.
4. **Knowledge Base Integration**: Perform rule-based reasoning on detected relationships.


## Setup Instructions

### Clone the Repository
To get started, clone this repository to your local machine:
```bash
git clone https://github.com/hassan-byt0/SceneGraph.git
cd SceneGraph
```

### Install Dependencies( Optional as this step is already covered in notebook)
Follow these steps to set up the required environment:
1. Install essential libraries:
    ```bash
    pip install torch torchvision
    pip install torch-geometric
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121
    ```

2. Clone the Detectron2 repository and install dependencies:
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    python -m pip install -r requirements.txt
    cd ..
    ```

3. Install additional dependencies for PyYAML:
    ```bash
    python -m pip install pyyaml==5.1
    ```

4. Add Detectron2 to Python's path (for Colab or local setups):
    ```python
    import sys, os
    sys.path.insert(0, os.path.abspath('./detectron2'))
    ```

---

## Running the Notebook
### Steps to Run
1. Launch the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open the provided notebook file, `scene_graph_reasoning.ipynb`.
3. Replace the placeholder in the code with the path to your image or use the sample image provided: `SceneGraph/car distance.webp`.
4. Run all the cells in the notebook sequentially to:
    - Detect objects in the image.
    - Visualize detection results.
    - Construct the scene graph.
    - Apply the GCN and perform knowledge-based reasoning.

---

## Example
### Input
An image containing multiple objects (e.g., a car, person, and bicycle). Replace the `image_path` with your own image or use the included `/SceneGraph/car distance.webp`.

### Output
- **Visualized Object Detection**: Displays bounding boxes and labels for detected objects.
- **Scene Graph Reasoning**: Outputs node embeddings and relationship insights using a GCN and predefined rules.

---

## Files
- `scene_graph_reasoning.ipynb`: The main notebook containing the implementation.
- `car distance.webp`: Sample image for testing.( your image)

---

## Future Work
- Expand the knowledge base with more complex rules.
- Enhance edge definitions with spatial and semantic features.
- Extend support for dynamic graphs with video data.
- Optimize the Graph Neural net

---

## Acknowledgments
This project leverages:
- [Detectron2](https://github.com/facebookresearch/detectron2) for object detection.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph processing.
- Jupyter Notebook for an interactive coding environment.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Contact
For queries or suggestions, contact [Hassan_Shaikh](shaikhhassan0502@gmail.com).

