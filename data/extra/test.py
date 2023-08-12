import cc3d
import numpy as np
x=np.random.randint(0,2,[5,7,6])
x[1]=0
x[3]=0
y,N=cc3d.connected_components(x ,return_N=True)
print ("N",N)
for label, image in cc3d.each(y, binary=False, in_place=True):
    print ("label",label)
    print ("unique",np.unique(image))

print ("after")
for index,(label, image) in enumerate(cc3d.each(y, binary=True, in_place=True)):
    assert index+1==label, f"index {index},label {label}"
    print ("label",label)
    print ("unique",np.unique(image))

