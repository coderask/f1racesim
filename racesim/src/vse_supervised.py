import pickle
import numpy as np
import tensorflow as tf
import os
import configparser
import ast
import pickle


class VSE_SUPERVISED(object):
    
    """
    author:
    Alexander Heilmeier

    date:
    02.04.2020

    .. description::
    This class provides handles two neural networks (and the according preprocessors) to take tire change decisions,
    i.e. the timing of the pit stop (tc = tirechange) as well as the compound decision (cc = compound choice), during
    races on the basis of the current situation. It provides four main methods: preprocessing of the data for tc and cc
    and taking the decision for tc and cc. Keep in mind that some of the inputs have to be in the state at the end of
    the previous lap!
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    
    __slots__ = ("__preprocessor_cc",
                 "__preprocessor_tc",
                 "__nnmodel_cc",
                 "__nnmodel_tc",
                 "__X_conv_cc",
                 "__X_conv_tc",
                 "__no_timesteps_tc",
                 "collected_data")


    #extracting required race file
     # laps = config['RACE_PARS']
    # for key in laps:
    #     for key1 in laps[key]:
    #       print(ast.literal_eval(key1))
    #print(laps)
    path = "racesim/input/parameters/pars_Catalunya_2019.ini"
    config = configparser.ConfigParser()
    config.read(path)
    driver_pars_section = config['DRIVER_PARS']
    global driver_pars_dict 
    driver_pars_dict= {}
    for key in driver_pars_section:
        driver_pars_dict[key] = ast.literal_eval(driver_pars_section[key])
    #print("driver_pars_dict:", driver_pars_dict['driver_pars']["HAM"])
    race_pars_section = config['RACE_PARS']
    race_pars_str = race_pars_section['race_pars']
    race_pars_str = race_pars_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    race_pars_dict = ast.literal_eval(race_pars_str)
    global tot_no_laps
    tot_no_laps = race_pars_dict['tot_no_laps']
    print(tot_no_laps)
    #intialzing and compiling model 
    global final_dataset
    final_dataset = {}
    global collected_data
    collected_data = {}
    global prev_collected_data 
    prev_collected_data = {}
    global driver_order 
    driver_order = [None] * 20
    global driver_compound_choices
    driver_compound_choices = {}

    with open('racesim/src/ccModelRecreation/lapdict.pkl', 'rb') as pikle:
        lapdict = pickle.load(pikle)
    print(lapdict)

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 preprocessor_cc_path: str,
                 preprocessor_tc_path: str,
                 nnmodel_cc_path: str,
                 nnmodel_tc_path: str) -> None:

        with open(preprocessor_cc_path, 'rb') as fh:
            self.preprocessor_cc = pickle.load(fh)

        with open(preprocessor_tc_path, 'rb') as fh:
            self.preprocessor_tc = pickle.load(fh)

        self.nnmodel_cc = {"interpreter": tf.lite.Interpreter(model_path=nnmodel_cc_path)}
        self.nnmodel_tc = {"interpreter": tf.lite.Interpreter(model_path=nnmodel_tc_path)}

        self.X_conv_cc = None
        self.X_conv_tc = None
        # initialize tf lite interpreters
        self.nnmodel_cc["interpreter"].allocate_tensors()
        self.nnmodel_cc["input_index"] = self.nnmodel_cc["interpreter"].get_input_details()[0]['index']
        self.nnmodel_cc["output_index"] = self.nnmodel_cc["interpreter"].get_output_details()[0]['index']

        self.nnmodel_tc["interpreter"].allocate_tensors()
        self.nnmodel_tc["input_index"] = self.nnmodel_tc["interpreter"].get_input_details()[0]['index']
        self.nnmodel_tc["output_index"] = self.nnmodel_tc["interpreter"].get_output_details()[0]['index']

        self.no_timesteps_tc = self.nnmodel_tc["interpreter"].get_input_details()[0]['shape'][1]


        #aarnav added code
        #collect all IO from their regular use of tf-lite models
        #set up the race file


    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_preprocessor_cc(self): return self.__preprocessor_cc
    def __set_preprocessor_cc(self, x) -> None: self.__preprocessor_cc = x
    preprocessor_cc = property(__get_preprocessor_cc, __set_preprocessor_cc)

    def __get_preprocessor_tc(self): return self.__preprocessor_tc
    def __set_preprocessor_tc(self, x) -> None: self.__preprocessor_tc = x
    preprocessor_tc = property(__get_preprocessor_tc, __set_preprocessor_tc)

    def __get_nnmodel_cc(self) -> dict: return self.__nnmodel_cc
    def __set_nnmodel_cc(self, x: dict) -> None: self.__nnmodel_cc = x
    nnmodel_cc = property(__get_nnmodel_cc, __set_nnmodel_cc)

    def __get_nnmodel_tc(self) -> dict: return self.__nnmodel_tc
    def __set_nnmodel_tc(self, x: dict) -> None: self.__nnmodel_tc = x
    nnmodel_tc = property(__get_nnmodel_tc, __set_nnmodel_tc)

    def __get_X_conv_cc(self) -> np.ndarray: return self.__X_conv_cc
    def __set_X_conv_cc(self, x: np.ndarray) -> None: self.__X_conv_cc = x
    X_conv_cc = property(__get_X_conv_cc, __set_X_conv_cc)

    def __get_X_conv_tc(self) -> np.ndarray: return self.__X_conv_tc
    def __set_X_conv_tc(self, x: np.ndarray) -> None: self.__X_conv_tc = x
    X_conv_tc = property(__get_X_conv_tc, __set_X_conv_tc)

    def __get_no_timesteps_tc(self) -> int: return self.__no_timesteps_tc
    def __set_no_timesteps_tc(self, x: int) -> None: self.__no_timesteps_tc = x
    no_timesteps_tc = property(__get_no_timesteps_tc, __set_no_timesteps_tc)
    # ------------------------------------------------------------------------------------------------------------------
    # METHODS ----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        # reset VSE such that it can be used to simulate (same race) again
        self.X_conv_cc = None
        self.X_conv_tc = None

    def preprocess_features(self,
                            # TC -----------
                            tireageprogress_corr_zeroinchange: list,
                            raceprogress: float,
                            position: list,
                            rel_compound_num_nl: list,
                            fcy_stat_nl: list,
                            remainingtirechanges_nl: list,
                            tirechange_pursuer: list,
                            location_cat: int,
                            close_ahead_prevlap: list,
                            # CC -----------
                            location: str,
                            used_2compounds_nl: list,
                            no_avail_dry_compounds: int) -> None:

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPARATION (TIRE CHANGE DECISION) -------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # get number of drivers
        no_drivers_tmp = len(position)

        # set everything together
        X = np.zeros((no_drivers_tmp, 9))  # currently we have 9 input features
        X[:, 0] = tireageprogress_corr_zeroinchange
        X[:, 1] = raceprogress
        X[:, 2] = position
        X[:, 3] = rel_compound_num_nl
        X[:, 4] = fcy_stat_nl
        X[:, 5] = remainingtirechanges_nl
        X[:, 6] = tirechange_pursuer
        X[:, 7] = location_cat
        X[:, 8] = close_ahead_prevlap
        
        #print("position passed to VSE SUPERVISED: ", position)
        #print out all raw input data 
        #print("Raw input data for CC decision:")
        #print("raceprogress:", raceprogress)
        #print("location:", location)
        #print("rel_compound_num_nl:", rel_compound_num_nl)
        #print("remainingtirechanges_nl:", remainingtirechanges_nl)
        #print("used_2compounds_nl:", used_2compounds_nl)
        #print("no_avail_dry_compounds:", no_avail_dry_compounds)
        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPROCESSING (TIRE CHANGE DECISION) -----------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if self.X_conv_tc is None:
            # initialize X_conv_tc if called for the first time
            self.X_conv_tc = np.zeros((no_drivers_tmp, self.no_timesteps_tc, self.preprocessor_tc.no_transf_cols),
                                      dtype=np.float32)

            # set correct initial values for every driver
            for idx_driver in range(no_drivers_tmp):
                X_conv_tc_tmp = np.tile(X[idx_driver], (self.no_timesteps_tc, 1))

                # set FCY status 0 for every lap except lap 0
                X_conv_tc_tmp[:-1, 4] = 0

                # process features
                self.X_conv_tc[idx_driver] = self.preprocessor_tc.transform(X_conv_tc_tmp, dtype_out=np.float32)

        else:
            # process new features
            X_conv_tc_tmp = self.preprocessor_tc.transform(X, dtype_out=np.float32)

            # replace last entry in X_conv_tc for every driver by new data
            self.X_conv_tc = np.roll(self.X_conv_tc, -1, axis=1)
            self.X_conv_tc[:, -1] = X_conv_tc_tmp

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPARATION (COMPOUND CHOICE DECISION) ---------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        X = np.zeros((no_drivers_tmp, 6))  # currently we have 6 input features
        X[:, 0] = raceprogress
        X[:, 1] = self.preprocessor_cc.transform_cat_dict(X_cat_str=[location], featurename='location')
        X[:, 2] = rel_compound_num_nl
        X[:, 3] = remainingtirechanges_nl
        X[:, 4] = used_2compounds_nl
        X[:, 5] = no_avail_dry_compounds

        # --------------------------------------------------------------------------------------------------------------
        # FEATURE PREPROCESSING (COMPOUND CHOICE DECISION) -------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # process new features
        self.X_conv_cc = self.preprocessor_cc.transform(X, dtype_out=np.float32)
    

    def make_decision(self,
                      bool_driving: list or np.ndarray,
                      avail_dry_compounds: list,
                      param_dry_compounds: list,
                      remainingtirechanges_curlap: list,
                      used_2compounds: list,
                      cur_compounds: list,
                      raceprogress_prevlap: float,
                      position: list,
                      driver_intials: str) -> list:
        lapno = tot_no_laps * raceprogress_prevlap
        lapno = round(lapno, 0)
        #(lapno)
        #print(position)
        #rint(used_2compounds)
        #print("driver_intials:", driver_intials)
        # get number of drivers and create output list
        no_drivers_tmp = self.X_conv_tc.shape[0]
        next_compounds = [None] * no_drivers_tmp

        # --------------------------------------------------------------------------------------------------------------
        # TIRE CHANGE DECISION -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # create array for prediction probabilities
        pitstop_probs = np.zeros(no_drivers_tmp, dtype=np.float32)
        cur_dict=[]
        for idx_driver in range(no_drivers_tmp):
            #print("cur driver index:", idx_driver)
            #print(driver_intials)
            # continue if driver does not participate anymore
            if not bool_driving[idx_driver]:
                continue
            #following their preprocessing reqs
            if(position[idx_driver] <= 10):
                cur_dict.append({idx_driver : self.X_conv_cc[idx_driver]})
                collected_data[lapno] = cur_dict    
            # set NN input
            self.nnmodel_tc["interpreter"].set_tensor(self.nnmodel_tc["input_index"],
                                                      np.expand_dims(self.X_conv_tc[idx_driver], axis=0))
            #print(self.X_conv_tc[idx_driver])
            # invoke NN
            self.nnmodel_tc["interpreter"].invoke()

            # fetch NN output
            pitstop_probs[idx_driver] = self.nnmodel_tc["interpreter"].get_tensor(self.nnmodel_tc["output_index"])            
        # get indices of the drivers that have a predicted pitstop probability above 50%
        idxs_driver_pitstop = list(np.flatnonzero(np.round(pitstop_probs)))

        # assure that every driver used two different compounds in a race (as soon as a raceprogress of 90% is exceeded)
        if raceprogress_prevlap > 0.9 and not all(used_2compounds):
            for idx_driver in range(no_drivers_tmp):
                if not used_2compounds[idx_driver] \
                        and bool_driving[idx_driver] \
                        and idx_driver not in idxs_driver_pitstop:

                    idxs_driver_pitstop.append(idx_driver)
                    print("WARNING: Had to enforce a pit stop for supervised VSE above 90%% race progress (driver at"
                          " index %i of supervised drivers)!" % idx_driver)

            idxs_driver_pitstop.sort()

        # --------------------------------------------------------------------------------------------------------------
        # COMPOUND CHOICE DECISION -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # compound choice is only performed if any driver was chosen for a pit stop
        if idxs_driver_pitstop:
            #print("driver intials list:", driver_intials)
            #print("list of drivers chosen for pitstop:", idxs_driver_pitstop)

            # create array for prediction probabilities (NN was trained with 3 different compounds to choose from)
            rel_compound_probs = np.zeros((len(idxs_driver_pitstop), 3), dtype=np.float32)
            '''print("model INput")
            print(self.X_conv_cc)'''
            #print("model Input shape")z
            #print(self.X_conv_cc.shape)
            i = 0
            #cur_dict = []
            for idx_rel, idx_abs in enumerate(idxs_driver_pitstop):
                # print("drivers selected:", idxs_driver_pitstop)
                # print(i)
                # i+=1
                # set NN input
                #print("SET NN INPUT")
                self.nnmodel_cc["interpreter"].set_tensor(self.nnmodel_cc["input_index"],
                                                          np.expand_dims(self.X_conv_cc[idx_abs], axis=0))
                #print("index:", idx_abs)
                #print("input:", self.X_conv_cc[idx_abs])
                #print("\n")
                #print("expanded dims shape")
                #print(np.expand_dims(self.X_conv_cc[idx_abs], axis = 0).shape)
                #add input to the array
                #add normally if only one driver selected for pit
                # if len(idxs_driver_pitstop) == 1: 
                #     collected_data[lapno] = {idx_abs : self.X_conv_cc[idx_abs]}
                # else: 
                #     cur_dict.append({idx_abs : self.X_conv_cc[idx_abs]})
                #     collected_data[lapno] = cur_dict
                #newModel
                #print("")
                #print("")
                #print(idx_abs)
                #print("This is the input being passed to the Model:", np.expand_dims(self.X_conv_cc[idx_abs], axis=0) )
                #print("aarnav newmodel started intializing ")
                '''
                newModel = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(np.expand_dims(self.X_conv_cc[idx_abs],axis=0).shape[1],)),  # Input layer with the flattened shape
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(3, activation='softmax')
                ])          
                #print("newmodel compiling")
                newModel.compile(optimizer=tf.keras.optimizers.Nadam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
                #print("sucessful compile")

                #print("newmodel fit")

                input_data = np.expand_dims(self.X_conv_cc[idx_abs], axis=0)
                #print(f"Input data shape for idx_abs {idx_abs}: {input_data.shape}")
                
                # Ensure input data is a NumPy array
                labels = [0,0.15,0.85]
                input_data = np.array(input_data, dtype=np.float32)
                label = np.array([labels[idx_rel]], dtype=np.float32)

                # Fit the model
                newModel.fit(input_data, label, epochs=1, validation_data=(input_data, label))
                
                # Predict probabilities
                print("newModel preds. : ")
                print("    index: ", idx_rel)
                print("predicition:", newModel.predict(input_data))
                '''

                
                # invoke NN
                self.nnmodel_cc["interpreter"].invoke()
                
                # fetch NN output
                #print("")
                #print("this is the output of the network:")
                #print(rel_compound_probs)
                #print(self.nnmodel_cc["interpreter"].get_tensor(self.nnmodel_cc["output_index"]))
                rel_compound_probs[idx_rel] = \
                    self.nnmodel_cc["interpreter"].get_tensor(self.nnmodel_cc["output_index"])
                
                #print(rel_compound_probs[idx_rel])
                #print("their prediction: ", rel_compound_probs[idx_rel])
            # get array with indices of relative compounds sorted by highest -> lowest probability
            idxs_rel_compound_sorted = list(np.argsort(-rel_compound_probs, axis=1))

            # make sure that VSE chooses only from available compounds (only 2 different compounds were available before
            # 2016, some of the compounds might not be parameterized in the current race for some drivers) -> use the
            # compound with the next lower probability if chosen compound is not available
            for idx_rel, idx_abs in enumerate(idxs_driver_pitstop):

                # loop through relative compound indices from highest to lowest probability
                for idx_rel_compound in idxs_rel_compound_sorted[idx_rel]:

                    # case two available compounds: in 2014 and 2015 we have only medium and soft -> subtract 1 from
                    # decision to fit available compounds with indices 0 and 1
                    if len(avail_dry_compounds) == 2:
                        idx_rel_compound_corr = idx_rel_compound - 1

                        # continue to compound with next lower probability if current compound is not available in the
                        # race (index - 1) or not parameterized
                        if idx_rel_compound_corr < 0 \
                                or avail_dry_compounds[idx_rel_compound_corr] not in param_dry_compounds:
                            continue

                    # case three available compounds
                    else:
                        idx_rel_compound_corr = idx_rel_compound

                        # continue to compound with next lower probability if current compound is not parameterized
                        if avail_dry_compounds[idx_rel_compound_corr] not in param_dry_compounds:
                            continue

                    # continue to compound with next lower probability if this is the last planned pit stop (or if it is
                    # probably the last pit stop because 90% race progress are already exceeded) and if the driver would
                    # not drive two different compounds in the race if the current compound was chosen
                    if (remainingtirechanges_curlap[idx_abs] == 1 or raceprogress_prevlap > 0.9) \
                            and not used_2compounds[idx_abs] \
                            and avail_dry_compounds[idx_rel_compound_corr] == cur_compounds[idx_abs]:
                        continue

                    # set new compound
                    next_compounds[idx_abs] = avail_dry_compounds[idx_rel_compound_corr]
                    break
        #AARNAV MODEL START 
        
        #print("I to the model", self.collected_data)
        #driverchoices = [["A4", "A3", "A4"], ]
        return next_compounds
    def expData(self, pits, collected_data, dOrder, ac, tl, threePitIgnore):
        with open('racesim/src/expData/expDataCat2019.pkl', 'wb') as file:
            pickle.dump({'pits': pits, 'collected_data': collected_data, "driver_order" : dOrder, "avail" : ac, "totLaps" : tl, "threePitIgnore" : threePitIgnore}, file)
    # print(f"Data")
    def trainTyreModel(self,
                      bool_driving: list or np.ndarray,
                      avail_dry_compounds: list,
                      param_dry_compounds: list,
                      remainingtirechanges_curlap: list,
                      used_2compounds: list,
                      cur_compounds: list,
                      raceprogress_prevlap: float,
                      driver_intials: str,
                      positions: list):
        global prev_collected_data  
        #contains all the collected inputs to the cc model in a dictionary s.t {lapnumber : {driverno: input}}} or {lapnumber: [{driverno1:input}, {driverno2: input}]}
        global collected_data  
        global final_dataset  
        global driver_order
        lapno = tot_no_laps * raceprogress_prevlap
        #print("first", lapno)
        lapno = round(lapno, 0)
        print("lap:", lapno)
        #print(len(collected_data))
        #print(len(self.collected_data[1]))
        #print(i)
        #i+=1
        #define model
        
        # newModel = tf.keras.models.Sequential([
        #         tf.keras.layers.Input(shape=(34,)),  # Input layer with the flattened shape
        #         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        #         tf.keras.layers.Dense(3, activation='softmax')
        #         ]) 
        # #compile model
        # newModel.compile(optimizer=tf.keras.optimizers.Nadam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])       

        #print("driver initals given to trainTyreMOdel:", driver_intials)
        #print("lastlap_positions:", positions)
        #train model 
        #convert position and name into a dicitonary 
        newDict = {}
        driver_pit_lap_nos = {}
        pit_lap_nos = []
        i=0
        #print(lapno)
        # print(len(self.collected_data))
        
        # print("\n")
        for pos in positions:
            newDict[driver_intials[i]] = pos
            i+=1
        # for label in newDict:
        #      print(label)
        #      print(driver_pars_dict['driver_pars'][label]['strategy_info'])
        #      print("\n")
        #      pass

        #creates a dictionary with pit_no:pitlap, pit_no_2:pitlap, pit_no_3:pitlap
        ign_dri_init = []
        for driver_idx,label in enumerate(newDict):
            #print(label)
            #print(driver_idx)
            for lapindex in driver_pars_dict["driver_pars"][label]["strategy_info"]:
                if driver_pit_lap_nos.get(label) is None:
                    driver_pit_lap_nos[label] = (lapindex[0],lapindex[1])
                    # driver_compound_choices[label] = lapindex[1]
                    pit_lap_nos.append(lapindex[0])
                elif driver_pit_lap_nos.get(label + "_2") is None: 
                    tmp_label = label + "_2"
                    driver_pit_lap_nos[tmp_label] = (lapindex[0], lapindex[1])
                    pit_lap_nos.append(lapindex[0])
                elif driver_pit_lap_nos.get(label+ "_3") is None: 
                    driver_pit_lap_nos[label + "_3"] = (lapindex[0], lapindex[1])
                    pit_lap_nos.append(lapindex[0])
                else: 
                    ign_dri_init.append(label)
                    #print('DRIVER OVER 3 PITSTOPS REFER BACK TO 511 VSE SUPERVISED')
                #print(lapindex[0], ":", lapindex[1])
                #print(lapindex[1])
            #print(self.collected_data)
            #print("\n")
        #list of laps where a pitstop occured
        pit_lap_nos = set(pit_lap_nos)
        #print(pit_lap_nos)
       #print(driver_pit_lap_nos)    
        


        #deciphering index of inputs
        #uses only lap 0 because that matches grid positions 
        if lapno == 0.0:
            #consider each entry of positions(follows same indexing as self.X_conv_tc/cc)
            for idx, pos in enumerate(positions):
                for label in driver_pars_dict["driver_pars"]:
                    #print("p_grid:", driver_pars_dict["driver_pars"][label]["p_grid"])
                    if driver_pars_dict["driver_pars"][label]['p_grid'] == pos:
                        driver_order[idx] = label
        #print("Driver order:", driver_order)


        '''create data'''
        # if len(prev_collected_data) != collected_data:
        #     for driver in driver_pit_lap_nos:
        #         #print("Driver:", driver)
        #         if lapno == driver_pit_lap_nos[driver]:
        #             #print("pitlap for", driver)
        #             pass
        #     prev_collected_data = collected_data
        #     final_dataset[lapno] = collected_data
        # else:
        #     final_dataset[lapno] = prev_collected_data
        # print(newDict)
        # print(lapno)
        # print(driver_pit_lap_nos)
        # print(driver_order)
        if lapno == (tot_no_laps-1):
            #print(final_dataset[65])
            #print(collected_data)
            #print(driver_pit_lap_nos)
            print("test")
            self.expData(driver_pit_lap_nos, collected_data, driver_order, avail_dry_compounds, tot_no_laps, ign_dri_init)
            

# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass