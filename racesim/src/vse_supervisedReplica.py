import pickle 

# newModel = tf.keras.models.Sequential([
        #         tf.keras.layers.Input(shape=(34,)),  # Input layer with the flattened shape
        #         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #         tf.keras.layers.Dense(3, activation='softmax')
        #         ]) 
        # #compile model
        # newModel.compile(optimizer=tf.keras.optimizers.Nadam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])       

        #print("driver initals given to trainTyreMOdel:", driver_intials)
        #print("lastlap_positions:", positions)
#load data in 
with open('/Users/aarnavkoushik/Documents/GitHub/f1racesim/racesim/src/expData/expDataCat2019.pkl', 'rb') as file:
    data = pickle.load(file)
pits = data['pits']
inp = data['collected_data']

print(pits)
print("\n")
print(inp)
