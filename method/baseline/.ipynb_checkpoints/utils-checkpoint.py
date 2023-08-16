import pickle
import os

def save_var(v,filepath,filename):
    if not os.path.exists("/root/autodl-tmp/method/temp/"+filepath):
        os.mkdir("/root/autodl-tmp/method/temp/"+filepath)
    f=open("/root/autodl-tmp/method/temp/"+filepath+"/"+filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename


def load_var(filepath,filename):
    f=open("/root/autodl-tmp/method/temp/"+filepath+"/"+filename,'rb')
    r=pickle.load(f)
    f.close()
    return r