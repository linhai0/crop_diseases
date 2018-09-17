import numpy  as  np
import matplotlib.pyplot as plt
import cv2


# img = np.random.normal(0,1,3)

def abc(a=None, b=None, c=None, *args, **kwargs):
    print(a,b,c,args, kwargs)
abc(1,2,3, 3,4,5,m=1)
