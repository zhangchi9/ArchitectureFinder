import webapp.config as creds
import psycopg2
from numpy import dot
from numpy.linalg import norm

class_index = {0: 'Achaemenid architecture',
 1: 'American Foursquare architecture',
 2: 'American craftsman style',
 3: 'Ancient Egyptian architecture',
 4: 'Art Deco architecture',
 5: 'Art Nouveau architecture',
 6: 'Baroque architecture',
 7: 'Bauhaus architecture',
 8: 'Beaux-Arts architecture',
 9: 'Byzantine architecture',
 10: 'Chicago school architecture',
 11: 'Colonial architecture',
 12: 'Deconstructivism',
 13: 'Edwardian architecture',
 14: 'Georgian architecture',
 15: 'Gothic architecture',
 16: 'Greek Revival architecture',
 17: 'International style',
 18: 'Novelty architecture',
 19: 'Palladian architecture',
 20: 'Postmodern architecture',
 21: 'Queen Anne architecture',
 22: 'Romanesque architecture',
 23: 'Russian Revival architecture',
 24: 'Tudor Revival architecture'}

def connect():
    """
    Set up a connection to the postgres server, and return a cursor object.
    """ 
    conn_string = "host="+ creds.PGHOST +" port="+ "5432" +" dbname="+ creds.PGDATABASE +" user=" + creds.PGUSER \
                  +" password="+ creds.PGPASSWORD
    
    conn = psycopg2.connect(conn_string)
    print("Connected to AWS DataBase!")
    cursor = conn.cursor()
    
    return conn, cursor

def sketchify(jc):
    """
    Turn a image into a sketch
    input: 
        jc, numpy array
    output: 
        image, numpy array
    """
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

def cosine_similarity(a,b):
    """
    Calculate the cosine similarity between two vectors a and b
    inputs: 
        a, b: numpy array
    output:
        cosine_similarity between a and b
    """
    return dot(a, b)/(norm(a)*norm(b))

def get_embeddings(filepath):
    """
    This function will calculate the image embeddings of a given image file path
    input: 
        filepath: string
    outputs:
        class_name: list[string]
            possible styles
        embedding_vector: numpy array
            embedding vector of the input image
        style_mp: string
            most probable style
    """
    img = cv2.imread(filepath)
    img = cv2.resize(img,(256,256),interpolation = cv2.INTER_AREA)
    sketch_im = sketchify(img)/255

    output = model.predict(sketch_im[np.newaxis,:,:,:])
    class_ = output[1]
    embedding_vector = output[0][0]
    index = np.where(class_[0] > 0.0)[0]
    class_name = []
    for i in index:
        class_name.append(class_index[i])
    style_mp = class_index[np.argmax(class_[0])]
    return class_name, embedding_vector, style_mp