import pickle
import os

def save_var(v,filepath,filename):
    if not os.path.exists("temp/"+filepath):
        os.mkdir("temp/"+filepath)
    f=open("temp/"+filepath+"/"+filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename


def load_var(filepath,filename):
    f=open("temp/"+filepath+"/"+filename,'rb')
    r=pickle.load(f)
    f.close()
    return r