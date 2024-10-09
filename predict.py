import argparse
import torch
from torchvision import transforms
from PIL import Image
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to name mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model

# Image processing function
def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Predicting function
def predict(image_path, model, top_k=5):
    image = process_image(image_path)
    with torch.no_grad():
        output = model(image)
        probs, indices = torch.topk(output, top_k)
        probs = probs.exp()  
        classes = [model.class_to_idx[i] for i in indices[0].tolist()]
    
    return probs.tolist(), classes

def main():
    args = parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    probs, classes = predict(args.image_path, model, args.top_k)
   
    class_names = [cat_to_name[str(c)] for c in classes]
    
    print(f'Probabilities: {probs}')
    print(f'Classes: {class_names}')

if __name__ == '__main__':
    main()
