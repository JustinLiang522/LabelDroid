import torch
import pickle
from ResNet import ResNet
from Transformer import Transformer
from PIL import Image
from torchvision import transforms
from data_utils.build_vocab import Vocabulary
from gensim import models
from flask import Flask
from flask_restful import Resource, Api, reqparse
import argparse
from hashlib import sha256

parser = argparse.ArgumentParser()
parser.add_argument('--img', action='store_true')
args = parser.parse_args()

# load models
w2v =  models.keyedvectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocab_path = 'vocab.pkl'
model_path = 'best_model.ckpt'

app = Flask(__name__)
api = Api(app)

cache = {}

if args.img:
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

        # check cache 
        result = []
        for i, image in enumerate(images):
            sha = sha256(image).hexdigest()
            result.append(cache[sha] if sha in cache else sha)

        # caption uncached images
        images = [img for img in images if sha256(img).hexdigest() not in cache]

        if not images:
            return result

        # hash uncached images
        hashes = [sha256(img).hexdigest() for img in images]

        images = [transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])(Image.frombytes(mode='RGB', size=size, data=image)) for (image, size) in zip(images, sizes)]
        images = torch.stack(images, 0).to(device)

        features = encoder(images)
        sentence_ids = decoder.evaluate(features).cpu().numpy()

        captions = [' '.join(word_list[1: word_list.index('<end>')]) if '<end>' in word_list else '' for word_list in [[vocab.idx2word[i] for i in s] for s in sentence_ids]]

        # add to cache
        for i, hash in enumerate(hashes):
            cache[hash] = captions[i]

        for i, res in enumerate(result):
            if type(res) == type('') and res in cache:
                result[i] = cache[res]

        return result


    class ImageCaption(Resource):
        def post(self):
            parser = reqparse.RequestParser()
            parser.add_argument('images')
            parser.add_argument('sizes')
            args = parser.parse_args()
            return {'captions': image_cap(eval(args['images']), eval(args['sizes']))}


    api.add_resource(ImageCaption, '/img')


class WMD(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s1')
        parser.add_argument('s2')
        args = parser.parse_args()
        return {'sim': str(w2v.similarity(args['s1'], args['s2']))}


api.add_resource(WMD, '/wmd')


if __name__ == '__main__':
    app.run(debug=False, port=8888)
