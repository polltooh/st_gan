import read_proto as rp
from model3 import STGan as model

if __name__ == "__main__":
    model_param = rp.load_proto("model_proto3") 
    st_gan = model(model_param)
    st_gan.mainloop()

