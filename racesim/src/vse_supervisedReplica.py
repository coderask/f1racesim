import pickle 
import tensorflow as tf 
#load data in 
with open('/Users/aarnavkoushik/Documents/GitHub/f1racesim/racesim/src/expData/expDataCat2019.pkl', 'rb') as file:
    data = pickle.load(file)
pits = data['pits']
inp = data['collected_data']
order = data['driver_order']
# print(pits)
# print("\n")
# print(inp)
# print("\n")
# print(order)

#intialize the model
newModel = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(34,)),  # Input layer with the flattened shape
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(3, activation='softmax')
                ]) 
#compile model
newModel.compile(optimizer=tf.keras.optimizers.Nadam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])       

       

#loop through every lap within the input data
for lap in inp:
    print(lap)
    #loop through every driver for each lap 
    for driver in inp[lap]:
           #if multiple drivers are pitting, go through them one-by-one
           if isinstance(driver, dict):
                for driverIdx in driver: 
                     print(driver[driverIdx])
           else:
              print(inp[lap][driver]) 

            
                