import os
import argparse
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
import numpy as np

def check_input(path1, path2):
    
    if not os.path.exists(path1):
        raise ValueError(f'Path {path1} does not exist')
    if not os.path.exists(path2):
        raise ValueError(f'Path {path2} does not exist')
    if not os.path.isdir(path1):
        raise ValueError(f'Path {path1} is not a directory')
    if not os.path.isdir(path2):
        raise ValueError(f'Path {path2} is not a directory')
    
    for file in os.listdir(path1):
        if not file.endswith('.jpg') and not file.endswith('.png'):
            raise ValueError(f'File {file} is not an image')
        if not os.path.exists(os.path.join(path2, file)):
            raise ValueError(f'File {file} does not exist in {path2}')
    print('Input check passed')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Calculate the ID score of two folders of images')
    parser.add_argument('--path1', type=str, required=True, help='Path to the first folder of images')
    parser.add_argument('--path2', type=str, required=True, help='Path to the second folder of images')
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to save the logs')
    parser.add_argument('--save_detection', action='store_true', help='Save the detected faces')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index')
    
    args = parser.parse_args()
    
    check_input(args.path1, args.path2)
    
    device = 'cuda:{}'.format(args.gpu) if args.gpu is not None else 'cpu'
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(f'{args.log_dir}/id_score.txt', 'w') as f:
        f.write('file,score\n')
    
    if args.save_detection:
        os.makedirs(f'{args.log_dir}/{args.path1}_detection', exist_ok=True)
        os.makedirs(f'{args.log_dir}/{args.path2}_detection', exist_ok=True)
    
    # Load models 
    print('Loading models...')
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    
    sims = []
    
    for img in tqdm(os.listdir(args.path1)):
        img1 = Image.open(os.path.join(args.path1, img))
        img2 = Image.open(os.path.join(args.path2, img))
        
        if args.save_detection:
            save_path1 = os.path.join(f'{args.log_dir}/{args.path1}_detection', img)
            save_path2 = os.path.join(f'{args.log_dir}/{args.path2}_detection', img)
        else:
            save_path1 = None
            save_path2 = None
        
        # Detect faces
        face1 = mtcnn(img1, save_path=save_path1).to(device)
        face2 = mtcnn(img2, save_path=save_path2).to(device)
        
        if face1 is None or face2 is None:
            continue
        
        # Calculate embeddings
        emb1 = resnet(face1.unsqueeze(0)).detach().cpu().numpy() # [1, 512]
        emb2 = resnet(face2.unsqueeze(0)).detach().cpu().numpy() # [1, 512]
        
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        with open(f'{args.log_dir}/id_score.txt', 'a') as f:
            f.write(f'{img},{sim}\n')
            
        sims.append(sim)
        
    print(f'ID score: {np.mean(sims)}')
        

    