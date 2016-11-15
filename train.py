from model import STGan as model
import read_proto as rp

if __name__ == "__main__":
    model_param = rp.load_proto("model_proto") 
    st_gan = model(model_param)
    st_gan.mainloop()

