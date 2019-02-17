import argparse
import load_data
import model_fuc

parser = argparse.ArgumentParser(description = 'Predict the image using trained model')
parser.add_argument('input', nargs = '?', action = "store", default = "flowers/test/1/image_06752.jpg")
parser.add_argument('checkpoint', nargs = '?', action = "store", default = "./checkpoint.pth")
parser.add_argument('--top_k', dest = 'top_k', nargs = '?', action = "store", type = int, default = 3)
parser.add_argument('--category_names', dest = 'cat', nargs = '?', action = "store", default = 'cat_to_name.json')
parser.add_argument('--gpu', dest = 'gpu', nargs='?', action="store", default='GPU')

pa = parser.parse_args()
input_path = pa.input
checkpoint = pa.checkpoint
topk = pa.top_k
cat = pa.cat
gpu = pa.gpu

model =  model_fuc.load_model(path = checkpoint)
flower_name, prob = model_fuc.predict(input_path, model, topk, power = gpu, category_names = cat)
print(flower_name)
print(prob)





