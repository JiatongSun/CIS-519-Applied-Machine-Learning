import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from dataset import SuperTuxDataset
from save_and_load import load_model
from tqdm import tqdm

LABEL_=['background','kart','pickup','nitro','bomb','projectile']

def load_data(dataset_path, data_transforms=None, 
              num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path,data_transforms)
    return DataLoader(dataset, num_workers=num_workers, 
                      batch_size=batch_size, shuffle=True)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def predict(model, inputs, device='cpu'):
    inputs = inputs.to(device)
    logits = model(inputs)
    return F.softmax(logits, -1)

def draw_bar(axis, preds, labels=None):
    y_pos = np.arange(6)
    axis.barh(y_pos, preds, align='center', alpha=0.5)
    axis.set_xticks(np.linspace(0, 1, 10))

    if labels:
        axis.set_yticks(y_pos)
        axis.set_yticklabels(labels)
    else:
        axis.get_yaxis().set_visible(False)

    axis.get_xaxis().set_visible(False)

def visualize_predictions():
  
    model = load_model()
    model.eval()

    validation_image_path='data/valid/' #enter the path 

    dataset = SuperTuxDataset(image_path=validation_image_path)

    f, axes = plt.subplots(2, 6)

    idxes = np.random.randint(0, len(dataset), size=6)

    for i, idx in enumerate(idxes):
        img, label = dataset[idx]
        preds = predict(model, img[None], device='cpu').detach().cpu().numpy()

        axes[0, i].imshow(TF.to_pil_image(img))
        axes[0, i].axis('off')
        draw_bar(axes[1, i], preds[0], LABEL_ if i == 0 else None)

    plt.show()
    
def test_performance_val(): 
        """test Validation accuracy"""        
        model = load_model()
        model.eval()
        # model=CNNClassifier()
        correct = 0        
        validation_image_path='data/valid/' #enter the path 
        with torch.no_grad():
            for data, target in tqdm(load_data(validation_image_path),
                                     position=0):
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                # print(pred)
        validloader = load_data(validation_image_path)
        for i, data in tqdm(enumerate(validloader, 0),position=0):
            inputs, labels = data
            preds = predict(model, inputs, device='cpu')
            acc = accuracy(preds, labels)
        print("\nYour accuracy is ", acc)
    
if __name__ == '__main__':
    # visualize_predictions()
    test_performance_val()