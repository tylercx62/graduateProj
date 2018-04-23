import cv2
import numpy as np
import os
from scipy.misc import imread, imresize
import sys



# load image
def get_imgs_fn1(file_name, path, dataset):

    #os.mkdir('Output')
    ang_block = 7   #14
    ang_take = 7   #8
    crop_size = 224

    ang_sum = ang_take*ang_take
    dim = [ang_block, ang_block]

    img_2D = cv2.imread(path + file_name, -1)
    img_2D = img_2D[:,:,0:3]

    '''
    print("img_2D1",img_2D)
    print("img_2D1",len(img_2D))   #5250
    print("img_2D1",len(img_2D[0]))  #7574
    print("img_2D1",len(img_2D[0][0]))  # 3 channels
    print("img_2D1",img_2D.min())   # 0   ([0,0,0] as black)
    print("img_2D1",img_2D.max())   # 65536 as white
    print("img_2D1",img_2D[8][8])   # check random [4629,4990,5518]
    '''
    #sys.exit("=====tyler======")
    #img_2D = img_2D/65535.0
    img_2D = img_2D/255

    h = int(img_2D.shape[0] / dim[0])
    w = int(img_2D.shape[1] / dim[1])

    fullInput = np.zeros((dim[0],dim[1],h,w,3))
    for i in range(dim[0]):
        for j in range(dim[1]):
            img = img_2D[j::dim[1], i::dim[0], :]
            fullInput[i,j,:,:,:] = img
            
    #fullInput = fullInput[4:12, 4:12,:,:,:]
    # use original h,w
    # h = 3787/7 = 541, w = 2632/7= 376
    # h-w/2 = 82.5, h_crop = 82+376 = 458 
    fullInput = fullInput[0:7, 0:7,:,:,:]
    #fullInput = fullInput[0:7, 0:7,:,82:458,:]


    #fullInput = fullInput.reshape([64,h,w,3])
    fullInput = fullInput.reshape([ang_sum,h,w,3])
    #fullInput = fullInput.reshape([ang_sum,376,376,3])

    
    fullInput1 = []

    '''
    for i in range(49):
        fullInput1.append(imresize(fullInput[i], (224,224)))
        cc = np.rot90(np.array(imresize(fullInput[i], (224,224))))
        fullInput1.append(list(cc))
    '''


    # 1.0 fullInput1.append(imresize(fullInput[0], (224,224)))
    # 1.1 fullInput2 = fullInput2[22:425,22:425,:]
    # 1.2 fullInput2 = fullInput2[45:403,45:403,:]

    fullInput1.append(imresize(fullInput[0], (224,224)))
    fullInput1.append(imresize(fullInput[42], (224,224)))
    fullInput1.append(imresize(fullInput[6], (224,224)))
    fullInput1.append(imresize(fullInput[48], (224,224)))
    fullInput1.append(imresize(fullInput[0], (224,224))) 

    '''
    if dataset == 'train':   
        if  path == './image/Fabric/' or path == './image/Fur/' or path == './image/Leather/' or path == './image/Glass/' or path == './image/Wood/' or path == './image/Stone/' or path == './image/Plastic/' :
            fullInput1 = []
            fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
            fullInput2 = fullInput2[22:425,22:425,:]
            fullInput1.append(imresize(fullInput2, (224,224)))
            fullInput2 = np.repeat(np.repeat(imresize(fullInput[42], (224,224)),2, axis=0), 2, axis=1)
            fullInput2 = fullInput2[22:425,22:425,:]
            fullInput1.append(imresize(fullInput2, (224,224)))
            fullInput2 = np.repeat(np.repeat(imresize(fullInput[6], (224,224)),2, axis=0), 2, axis=1)
            fullInput2 = fullInput2[22:425,22:425,:]
            fullInput1.append(imresize(fullInput2, (224,224)))
            fullInput2 = np.repeat(np.repeat(imresize(fullInput[48], (224,224)),2, axis=0), 2, axis=1)
            fullInput2 = fullInput2[22:425,22:425,:]
            fullInput1.append(imresize(fullInput2, (224,224)))
            fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
            fullInput2 = fullInput2[22:425,22:425,:]
            fullInput1.append(imresize(fullInput2, (224,224)))
            print ("special condition...")
    if dataset == 'test': 
        fullInput1 = []
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[42], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[6], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[48], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        print ("special condition...")
    '''

    '''
    fullInput1.append(imresize(fullInput[0], (224,224)))
    fullInput1.append(imresize(fullInput[42], (224,224)))
    fullInput1.append(imresize(fullInput[6], (224,224)))
    fullInput1.append(imresize(fullInput[48], (224,224)))
    fullInput1.append(imresize(fullInput[0], (224,224))) 
    
    cc = np.rot90(np.array(imresize(fullInput[0], (224,224))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.array(imresize(fullInput[42], (224,224))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.array(imresize(fullInput[6], (224,224))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.array(imresize(fullInput[48], (224,224))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.array(imresize(fullInput[0], (224,224))))
    fullInput1.append(list(cc))

    cc = np.rot90(np.rot90(np.array(imresize(fullInput[0], (224,224)))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.rot90(np.array(imresize(fullInput[42], (224,224)))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.rot90(np.array(imresize(fullInput[6], (224,224)))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.rot90(np.array(imresize(fullInput[48], (224,224)))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.rot90(np.array(imresize(fullInput[0], (224,224)))))
    fullInput1.append(list(cc))
    
    cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput[0], (224,224))))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput[42], (224,224))))))
    fullInput1.append(list(cc)) 
    cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput[6], (224,224))))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput[48], (224,224))))))
    fullInput1.append(list(cc))
    cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput[0], (224,224))))))
    fullInput1.append(list(cc))  
    
    if  path == './image/Fabric/' or path == './image/Fur/' or path == './image/Leather/' or path == './image/Glass/' or path == './image/Wood/' or path == './image/Stone/' or path == './image/Plastic/' :
        fullInput1 = []
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[42], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[6], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[48], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        fullInput1.append(imresize(fullInput2, (224,224)))

        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.array(imresize(fullInput2, (224,224))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[42], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.array(imresize(fullInput2, (224,224))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[6], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.array(imresize(fullInput2, (224,224))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[48], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.array(imresize(fullInput2, (224,224))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.array(imresize(fullInput2, (224,224))))
        fullInput1.append(list(cc))

        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224)))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[42], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224)))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[6], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224)))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[48], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224)))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224)))))
        fullInput1.append(list(cc))

        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224))))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[42], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224))))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[6], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224))))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[48], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224))))))
        fullInput1.append(list(cc))
        fullInput2 = np.repeat(np.repeat(imresize(fullInput[0], (224,224)),2, axis=0), 2, axis=1)
        fullInput2 = fullInput2[22:425,22:425,:]
        cc = np.rot90(np.rot90(np.rot90(np.array(imresize(fullInput2, (224,224))))))
        fullInput1.append(list(cc))



        
        print ("special condition...")

    
    '''


    '''
    #Using patches, no imresize!!
    fullInput0 = []
    fullInput0.append(fullInput[0])
    fullInput0.append(fullInput[48])
    fullInput0.append(fullInput[6])
    fullInput0.append(fullInput[42])
    fullInput0.append(fullInput[24])
    #fullInput1 = []
    #fullInput1.append(list(np.array(fullInput0)[0, 76:300, 76:300, :]))
    
    #fullInput1 = []
    for i in range(5):
        for j in range (1):
            for z in range (1):
                w_s = j*76
                w_e = j*76+224
                h_s = z*76
                h_e = z*76+224
                fullInput1.append(list(np.array(fullInput0)[i, w_s:w_e, h_s:h_e, :]))
    
    #print ("np.array(fullInput1).shape", np.array(fullInput1).shape)
    '''

    
    '''
    #4 corners Combine into one image 

    fullInput1 = list(fullInput[0]) + list(fullInput[42])
    fullInput2 = list(fullInput[6]) + list(fullInput[48])
    
    fullInput1 = list(np.rot90(np.array(fullInput1)))
    fullInput2 = list(np.rot90(np.array(fullInput2))) 

    fullInput1 = fullInput1 + fullInput2
    fullInput1 = list(np.rot90(np.array(fullInput1)))
    fullInput1 = list(np.rot90(np.array(fullInput1)))
    fullInput1 = list(np.rot90(np.array(fullInput1)))

    #print ("hii", np.array(fullInput1).shape)
    fullInput0 = []
    fullInput0.append(imresize(fullInput1, (224,224)))
    fullInput = fullInput0
    #print (np.array(fullInput0).shape)

    '''

    '''
    #direct visualise will lose some colors
    from scipy.misc import toimage
    for i in range (2):
        toimage(fullInput1[i]).show()
        #fullInput2 = np.repeat(np.repeat(fullInput1[i],2, axis=0), 2, axis=1)
        #toimage(fullInput2).show()
        #print (np.array(fullInput2).shape)
        #cc = np.rot90(np.array(fullInput1[i]))
        #toimage(list(cc)).show()
    #toimage(fullInput1[99]).show()
    #toimage(fullInput0[0]).show()
    sys.exit("=============typer================")
    #toimage(imresize(fullInput[0], (224,224))).show()
    '''
    
    # open this if not combine
    fullInput = fullInput1
    return fullInput


