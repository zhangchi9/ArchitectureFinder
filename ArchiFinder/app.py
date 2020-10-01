import streamlit as st
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from streamlit import caching



def load_embbedings():
    df = pd.read_csv('./embeddings.csv')
    return df

@st.cache
def sketchify(jc):
    jc = np.uint8(jc)
    scale_percent = 1
    width = int(jc.shape[1]*scale_percent)
    height = int(jc.shape[0]*scale_percent)
    dim = (width,height)
    resized = cv2.resize(jc,dim,interpolation = cv2.INTER_AREA)
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(resized,-1,kernel_sharpening)
    gray = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)
    inv = 255-gray
    gauss = cv2.GaussianBlur(inv,ksize=(15,15),sigmaX=0,sigmaY=0)
    def dodgeV2(image,mask):
        return cv2.divide(image,255-mask,scale=256)

    pencil_jc = 255 - dodgeV2(gray,gauss)
    img = np.zeros(jc.shape)
    img[:,:,0] = pencil_jc
    img[:,:,1] = pencil_jc
    img[:,:,2] = pencil_jc
    return img

@st.cache
def generate_embbedings(embedding_model, img):
    img = cv2.resize(img,(256,256),interpolation = cv2.INTER_AREA)
    sketch_im = sketchify(img)/255
    embeddings = embedding_model.predict(sketch_im[np.newaxis,:,:,:])
    return embeddings[0]

def cosine_similarity(a,b):
    return dot(a, b)/(norm(a)*norm(b))

@st.cache
def load_pretrained_model():
    weigths_path = '../../pretrained/sketch_embedding.h5'
    return load_model(weigths_path)

def main():
    #caching.clear_cache()
    st.title("Find the real architecture from sketch")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    embedding_model = load_pretrained_model()
    
    df = load_embbedings()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg",'png'])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        if len(image.shape) == 3 and  image.shape[2] == 4:
            image = image[:,:,0:3]
        if len(image.shape) == 2:
            image = np.stack([image,image,image],-1)

        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        embedding_vector = generate_embbedings(embedding_model, image)

        st.write("searching in the database...")
        my_bar = st.progress(0)
        similarity = []
        for i in range(len(df)):
            similarity.append(cosine_similarity(df.iloc[i,2:],embedding_vector))
            percent_complete =  (i+1)/len(df)
            my_bar.progress(percent_complete)
        df['similarity'] = similarity
        df = df.sort_values(by = 'similarity', ascending=False).iloc[:5,:]

        topsimilars = df.filepath.to_list()
        for filename in topsimilars:
            recommended_img = cv2.imread(f'{filename}')
            recommended_img = cv2.resize(recommended_img,(1024,1024),interpolation = cv2.INTER_AREA)
            st.image(recommended_img, use_column_width=True)


if __name__ == "__main__": 
    main()