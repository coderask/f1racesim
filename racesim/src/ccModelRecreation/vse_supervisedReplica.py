import pickle 
import tensorflow as tf 
import numpy as np
import os

'''DATA INTIALIZATION'''
#load data in 
with open("racesim/src/expData/expDataCat2019.pkl", 'rb') as file:
    data = pickle.load(file)
#dictionary with {driver intials: (lapnumber, compound)}
pits = data['pits']
#collected X_conv_cc stored as {lapnumber : {driver : input}} or for multiple drivers {lapnumber : [{driver1, input1}, {driver2, input2}]}
inp = data['collected_data']
#order of drivers as list of intials 
order = data['driver_order']
#available compounds 
avail = data["avail"]
#total laps 
tLaps = data["totLaps"]
# print(tLaps)
print(pits)
# print("\n")
# print(inp)
# print("\n")
# print(order)
#check if inputs already exist, if not make blank ones
if os.path.exists('racesim/src/ccModelRecreation/inputs.pkl'):
    with open('racesim/src/ccModelRecreation/inputs.pkl', 'rb') as file:
        trunc_inp = pickle.load(file)
        #print("inputed input", trunc_inp)
else:
    trunc_inp = []

if os.path.exists('racesim/src/ccModelRecreation/labels.pkl'):
    with open('racesim/src/ccModelRecreation/labels.pkl', 'rb') as file:
        label = pickle.load(file)
        #print(f"inputed label = {label}")
else:
    label = []
# trunc_inp = []
# label = []
# print(avail)
# trunc_inp = []
# label = []

'''ENCODE TIRE COMPOUND'''
def encode_compounds(compound):
    max = avail[2][1]
    mid = avail[1][1]
    low = avail[0][1]
    print("avail", avail)
    print("max", max)
    print("mid", mid)
    print("low", low)
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
''' this raised an error regarding using Sparse Categorical Cross Entropy instead of Categorial Cross Entropy'''
# #intialize the model
# newModel = tf.keras.models.Sequential([
#                 tf.keras.layers.Input(shape=(34,)),  # Input layer with the flattened shape
#                 tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#                 tf.keras.layers.Dense(3, activation='softmax')
#                 ]) 
# #compile model
# newModel.compile(optimizer=tf.keras.optimizers.Nadam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])       

newModel = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(34,)),  # Input layer with the flattened shape
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(3, activation='softmax')
                ]) 
newModel.compile(optimizer=tf.keras.optimizers.Nadam(), 
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
     #    print("driverIdx:", driverIdx)
        # print(driver)
     #    print("d", driverIdx)
        # go through all the real pit data
        for driverInitPit in pits:
          #find the inital corresponding to the current driver
          if order[driverIdx] in driverInitPit:
               # print("driverInitPit:", driverInitPit)
               #is cur_lap same as pit and not start of race 
               if lap == pits[driverInitPit][0] and lap != 0:
                   print(encode_compounds(pits[driverInitPit][1]))
                   #is the driver still driving
                #    if encode_compounds(pits([driverInitPit][1])) is not None:
                #        trunc_inp.append(driver[driverIdx])
                #        label.append(encode_compounds(pits[driverInitPit][1]))
               #     print(pits[driverInitPit][1])
               #     print(encode_compounds(pits[driverInitPit][1]))

                   

                   #print(f"{driver[driverIdx]} is the input on {lap} for {order[driverIdx]} pitstop and {encode_compounds(pits[driverInitPit][1])} is the label")
               #     print(inp[lap][driver][])
                    #trunc_inp.append(inp[lap][driverIdx])
                    #print(f"{driverInitPit} corresponding input is {inp[lap]}")
                    # print(pits[driverInitPit][1])
                    # print(encode_compounds(pits[driverInitPit][1]))
                    #label.append(encode_compounds(pits[driverInitPit][1]))
                #check which pitstop this is
                # print("1")
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


#save inputs for this race instance 
# with open('racesim/src/ccModelRecreation/inputs.pkl', 'wb') as file:
#     pickle.dump(trunc_inp, file)
# with open('racesim/src/ccModelRecreation/labels.pkl', 'wb') as file:
#     pickle.dump(label, file)

# print(trunc_inp)
# print(label)
# print(label)
# print(len(trunc_inp))       
# print(len(label))


# input = np.array(trunc_inp)
# # print(input)
# labels = np.array(label)
# # print(labels)
# input = input.reshape(input.shape[0], -1)


# newModel.load_weights("racesim/src/ccModelRecreation/ccShanghai2019.weights.h5")
# '''running 48 epochs on catulyna stablizes at 65% ish, more epochs doesnt seem to help'''
# newModel.fit(input, labels, epochs=6, validation_data=(input, labels))
# newModel.save_weights("racesim/src/ccModelRecreation/ccShanghai2019.weights.h5")