import pptk
import pickle
import numpy as np


if __name__ == '__main__':
    # pc = pickle.load(open(r"D:\sick\api\sick_visionary_samples\VisionaryToPointCloud\world_coordinates38987.pickle", "rb"))
    pc = np.load(open('../todel.np', 'rb'))

    v = pptk.viewer(pc.reshape(-1, 3))
