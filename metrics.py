import nibabel as nib
import sys
import os
from os import listdir
from os import path
from os.path import isfile,join
import medpy.metric.binary as mmb
import numpy

def do(argv):

    segResults_path = argv[0]
    references_path = argv[1]
    num_classes = argv[2]

    results = listdir(segResults_path)
    results.sort(key=lambda f: int(filter(str.isdigit, f)))
    references = listdir(references_path)
    references.sort(key=lambda f: int(filter(str.isdigit, f)))

    pack = zip(results,references)
#    print(pack)

    res = 1 # image-specific variable
    for resultname, referencename in pack:
        result = nib.load(join(segResults_path,resultname)).get_data()
        reference = nib.load(join(references_path,referencename)).get_data()

        reference = numpy.expand_dims(reference,axis=3)
        reference = numpy.expand_dims(reference,axis=4)

        print("ASSD: ", mmb.assd(result,reference,voxelspacing=res))

        Assd_array = []
        Dice_array = []
        print(num_classes)
        for c_i in xrange(0,int(num_classes)):
            dc = mmb.dc(result == c_i, reference == c_i)
            assd = mmb.assd(result == c_i, reference == c_i,voxelspacing=res)
            Assd_array.append(assd)
            Dice_array.append(dc)

        for i in xrange(0, len(Dice_array)):
            print('Dice score of class_{}'.format(i), Dice_array[i])

        for i in xrange(0,len(Assd_array)):
            print('ASSD score  of class_{}'.format(i), Assd_array[i])



        #print("obj_assd: ",obj_assd(result,reference))
        #print("ASD: ", asd(result,reference))
        #print("ASD: ", asd(reference,result))
        #print("HD: ", hd(result,reference))
        print("Dice: ", mmb.dc(result,reference))
        #print("JC: ",jc(result,reference))
if __name__ == '__main__':
  do(sys.argv[1:])
