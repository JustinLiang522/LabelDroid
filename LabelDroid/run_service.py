import torch
import pickle
from ResNet import ResNet
from Transformer import Transformer
from PIL import Image
from torchvision import transforms
from data_utils.build_vocab import Vocabulary
from gensim import models
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask_restful import Resource, Api, reqparse

# load models
# wmd = models.KeyedVectors.load('w2v-googleplay.model')
w2v =  models.keyedvectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
use = hub.load("use/universal-sentence-encoder_4")

vocab_path = 'vocab.pkl'
model_path = 'best_model.ckpt'

app = Flask(__name__)
api = Api(app)

# Device configuration
device = torch.device('cpu')

# Load vocabulary
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Load model
checkpoint = torch.load(model_path, map_location=device)

encoder = ResNet()
decoder = Transformer()

decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)

encoder.cpu()
decoder.cpu()

encoder.eval()
decoder.eval()


def image_cap(images, sizes):

    """image captioning"""
    images = [transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(Image.frombytes(mode='RGB', size=size, data=image)) for (image, size) in zip(images, sizes)]
    images = torch.stack(images, 0).to(device)

    features = encoder(images)
    sentence_ids = decoder.evaluate(features).cpu().numpy()

    return [' '.join(word_list[1: word_list.index('<end>')]) if '<end>' in word_list else '' for word_list in [[vocab.idx2word[i] for i in s] for s in sentence_ids]]


class WMD(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s1')
        parser.add_argument('s2')
        args = parser.parse_args()
        return {'distance': wmd.wv.wmdistance(eval(args['s1']), eval(args['s2']))}


api.add_resource(WMD, '/wmd')


class ImageCaption(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('images')
        parser.add_argument('sizes')
        args = parser.parse_args()
        return {'captions': image_cap(eval(args['images']), eval(args['sizes']))}


api.add_resource(ImageCaption, '/img')


class USE(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s1')
        parser.add_argument('s2')
        args = parser.parse_args()
        x, y = use([" ".join(eval(args['s1'])), " ".join(eval(args['s2']))])
        return {'sim': str(cosine_similarity([x], [y])[0][0])}


api.add_resource(USE, '/use')


class W2V(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s1')
        parser.add_argument('s2')
        args = parser.parse_args()
        return {'sim': str(w2v.similarity(args['s1'], args['s2']))}


api.add_resource(W2V, '/w2v')


if __name__ == '__main__':
    app.run(debug=False, port=8888)
