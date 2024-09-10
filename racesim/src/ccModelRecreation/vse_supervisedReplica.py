import pickle 
import tensorflow as tf 
import numpy as np
import os
# from pathlib import path 
'''DATA INTIALIZATION'''
#load data in 
with open("racesim/src/expData/expDataCat2019.pkl", 'rb') as file:
    data = pickle.load(file)
with open("racesim/src/ccModelRecreation/driver_idx_t10.pkl", 'rb') as file:
    driver_idx_t10 = pickle.load(file)
with open("racesim/src/ccModelRecreation/SqlOrder.pkl", 'rb') as file:
    sqlOrder = pickle.load(file)
with open("racesim/src/expData/savedIndexMapping.pkl", 'rb') as file:
    indexMap = pickle.load(file)
if(os.path.exists('racesim/src/ccModelRecreation/ignoreLongLaps.pkl')):
    with open('racesim/src/ccModelRecreation/pitquery.pkl') as file: 
        ignoreLongPits = pickle.load(file)
else:
    print("LONGPITS DNE ")
if(os.path.exists('racesim/src/ccModelRecreation/lapquery.pkl')):
    with open('racesim/src/ccModelRecreation/lapquery.pkl') as file:
        ignoreLongLaps = pickle.load(file)
else:
    print("LONGLAPs DNE ")
#dictionary with {driver intials: (lapnumber, compound)}
pits = data['pits']
#collected X_conv_cc stored as {lapnumber : {driver : input}} or for multiple drivers {lapnumber : [{driver1, input1}, {driver2, input2}]}
inp = data['collected_data']
# print(len(inp))
#order of drivers as list of intials 
order = data['driver_order']
# print(order)
#available compounds 
avail = data["avail"]
#total laps 
tLaps = data["totLaps"]
#drivers to ignore because they did too many pitstops 
ignPitThree = data["threePitIgnore"]

#mapping of vse:sql driver indicies
# indexMap = data["sqlOrderMap"]
# print("total laps", tLaps)
# print(pits)
# # print("\n")
# print(inp[1])
# print("\n")
# print(order)
# print(driver_idx_t10)
# print("indexMap", indexMap)
# print(ignPit) 
# print(ignoreLongLaps) 
# print(ignoreLongPits)
# check if inputs already exist, if not make blank ones
if os.path.exists('a racesim/src/ccModelRecreation/inputs.pkl'):
    with open('racesim/src/ccModelRecreation/inputs.pkl', 'rb') as file:
        trunc_inp = pickle.load(file)
        # trunc_inp = []
        #print("inputed input", trunc_inp)
else:
    trunc_inp = []

if os.path.exists('a racesim/src/ccModelRecreation/labels.pkl'):
    with open('racesim/src/ccModelRecreation/labels.pkl', 'rb') as file:
        label = pickle.load(file)
        # label = []
        #(f"inputed label = {label}")
else:
    label = []
# trunc_inp = []
# label = []
# print(avail)
# trunc_inp = []
tc_label = []
tc_input = []
'''ENCODE TIRE COMPOUND'''
def encode_compounds(compound):
    max = avail[2][1]
    mid = avail[1][1]
    low = avail[0][1]
    # print("avail", avail)
    # print("max", max)
    # print("mid", mid)
    # print("low", low)
    if compound[1] == max:
        return [0, 0, 1]
    elif compound[1] == mid:
        return [0, 1, 0]
    elif compound[1] == low:
        return [1, 0, 0]
#     if avail[2] == 'A4': 
#          if compound[1] == '4':
#               return [0, 0, 1] 
#          elif compound[1] == '3':
#               return [0, 1, 0]
#          else:
#               return [1, 0, 0]
#     elif avail[2] == 'A3': 
#          if compound[1] == 3:
#               return [0, 0, 1] 
#          elif compound[1] == 2:
#               return [0, 1, 0]
#          else:
#               return [1, 0, 0]
# print(pits["HAM"][1])
# print(encode_compounds(pits["HAM"][1]))



'''MODEL CREATION '''


#pitstop choice model #no LSTM layer currently


# pitModel = tf.keras.models.Sequential([
#     tf.keras.layers.Input(shape=(34,)),  # Input layer with the flattened shape
#     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
#     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
#     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# pitModel.compile(optimizer=tf.keras.optimizers.Nadam(),
#                  loss=tf.keras.losses.BinaryCrossentropy(),
#                  metrics=[tf.keras.metrics.BinaryAccuracy()])



#compound choice model 
ccModel = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(34,)),  # Input layer with the flattened shape
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(3, activation='softmax')
                ]) 
ccModel.compile(optimizer=tf.keras.optimizers.Nadam(), 
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), #changed from logits and Spare entropy
                 metrics=[tf.keras.metrics.CategoricalAccuracy()])

'''MODEL TRAINING'''
#loop through every lap
for lap in inp:
    lap = round(lap, 0)
    #loop through every driver for each lap
    for driver in inp[lap]:
      #print(driver)
      #go through drivers one-by-one
      for driverIdx in driver: 
        # print("driverIdx:", driverIdx)
        # print(driver)
     #    print("d", driverIdx)
        # go through all the real pit data
        for driverInitPit in pits:
          #map checking 
          for mapped_pair in indexMap:
            # print(driverInitPit, mapped_pair)
            if driverIdx == mapped_pair[0]:
                # print("enter map pair checking")
                if mapped_pair[1] in driver_idx_t10:
                    
                    # print(f"driver idx {mapped_pair[1]} found in driver idx t10")
        #   print(pits[driverInitPit])
          #find the inital corresponding to the current driver
          #if the inital is to be ignored due to 3 pitstops, pass over it 
                    if order[driverIdx] in ignPitThree:
                        # print(lap)
                        print("entred 3 pit ignore for: ", order[driverIdx])
                        pass
                    else: 
                        #find the inital corresponding to the current driver
                        #print(driverInitPit)
                        if order[driverIdx] in driverInitPit:
                            # print(lap)
                            # print("entred loop for:", driverInitPit)
                            # print("driverInitPit:", driverInitPit)
                            #is cur_lap same as pit and not start of race 
                            if lap == pits[driverInitPit][0] and lap != 0:
                                # print(pits[driverInitPit])
                                # print(driverInitPit, lap, driverIdx, mapped_pair[1])
                                print(lap,driverInitPit,mapped_pair[1],driverIdx)
                                trunc_inp.append(driver[driverIdx])
                                label.append(encode_compounds(pits[driverInitPit][1]))
                #     print(pits[driverInitPit][1])
               #     print(encode_compounds(pits[driverInitPit][1])
                   #print(f"{driver[driverIdx]} is the input on {lap} for {order[driverIdx]} pitstop and {encode_compounds(pits[driverInitPit][1])} is the label")
               #     print(inp[lap][driver][])
                    #trunc_inp.append(inp[lap][driverIdx])
                    #print(f"{driverInitPit} corresponding input is {inp[lap]}")
                    # print(pits[driverInitPit][1])
                    # print(encode_compounds(pits[driverInitPit][1]))
                    #label.append(encode_compounds(pits[driverInitPit][1]))
                #check which pitstop this is
                # print("1")
                #if the cur_lap is not a predicted pit lap 
            #    else:
            #        tc_input.append(driver[driverIdx])
            #        tc_label.append(0)
               #  if "_3" in driverInitPit:
               #      pass
               #      # print("1")
               #      # print(pits[driverInitPit][0])
               #      #is the current lap the same as the pit 
               #      if lap == pits[driverInitPit][0]:
               #          pass
               #            # print("1")
               #            # print (f"{order[driverIdx]} second pit on lap {pits[driverInitPit][0]} to {pits[driverInitPit][1]}")
               #  elif "_2" in driverInitPit:
               #      if lap == pits[driverInitPit][0]:
               #          pass
               #            # print (f"{order[driverIdx]} first pit on lap {pits[driverInitPit][0]} to {pits[driverInitPit][1]}")
               #  else: 
               #      if lap == pits[driverInitPit][0]:
               #            pass
               #            # print (f"{order[driverIdx]} start on lap {pits[driverInitPit][0]} with {pits[driverInitPit][1]}")
# #remove drivers who arent participating anymore 
# for inputInd, labelInd in enumerate(trunc_inp):
#     print(inputInd)
#     print(labelInd)


# thought there was a problem with loop so rewrote
# for lap, find driver input dict, for driver input dict, find driver index, for driver index, check if driver pits, if driver pits append. 
# for lap in inp: 
#     for driver in inp[lap]:
#         for driverIdx in driver:
#             for driverInitPit in pits:
                
#                 if order[driverIdx]+"_3" in driverInitPit:
#                     if lap == pits[driverInitPit][0] and lap != 0 and lap<32:
#                         print(lap, order[driverIdx],driverInitPit, pits[driverInitPit][0])
#                         # print(order[driverIdx], lap, pits[driverInitPit][1])
#                         trunc_inp.append(driver[driverIdx])
#                         label.append(encode_compounds(pits[driverInitPit][1]))
#                 elif order[driverIdx]+"_2" in driverInitPit:
#                     if lap == pits[driverInitPit][0] and lap != 0 and lap<32:
#                         print(lap, order[driverIdx],driverInitPit, pits[driverInitPit][0])
#                         # print(lap, order[driverIdx]+"_2", driverInitPit)
#                         trunc_inp.append(driver[driverIdx])
#                         label.append(encode_compounds(pits[driverInitPit][1]))
#                 elif order[driverIdx] in driverInitPit:
#                     # print(lap, order[driverIdx],driverInitPit, pits[driverInitPit][0])
#                     if lap == pits[driverInitPit] and lap!= 0 and lap<32:
#                         print(lap, order[driverIdx],driverInitPit, pits[driverInitPit][0])
#                         # print(lap, order[driverIdx], driverInitPit)
#                         trunc_inp.append(driver[driverIdx])
#                         label.append(encode_compounds(pits[driverInitPit][1]))
#                 else:
#                     pass
#                     # print("after third pit:", order[driverIdx])
                        


#collect data for tyre training model 

# save inputs for this race instance 
with open('racesim/src/ccModelRecreation/inputs.pkl', 'wb') as file:
    pickle.dump(trunc_inp, file)
with open('racesim/src/ccModelRecreation/labels.pkl', 'wb') as file:
    #print("saved label", label)
    pickle.dump(label, file)

# print(trunc_inp)
print("label", label)
# print(tc_label)
# print(len(trunc_inp))          
print(len(label))
# print(len(tc_label))
# print(len(tc_input))

# input = np.array(trunc_inp)
# # print(input)
# labels = np.array(label)
# # print(labels)
# input = input.reshape(input.shape[0], -1)


# ccModel.load_weights("racesim/src/ccModelRecreation/ccShanghai2019.weights.h5")
# '''running 48 epochs on catulyna stablizes at 65% ish, more epochs doesnt seem to help'''
# ccModel.fit(input, labels, epochs=6, validation_data=(input, labels))
# ccModel.save_weights("racesim/src/ccModelRecreation/ccShanghai2019.weights.h5")