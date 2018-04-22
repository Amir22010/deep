# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 01:10:25 2018

@author: Amir
"""


import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint
import matplotlib.pyplot as plt
import cnn_model



def main():
    
    
    model = cnn_model.fnBuildModel()
    
    print(model.summary())
    
    # save model to json
    model_json = model.to_json()
    with open('face_model.json', 'w') as json_file:
        json_file.write(model_json)
    
   
"""    X_train = np.load('data/X_train_train.npy')
    Y_train = np.load('data/Y_train_train.npy')
    X_test = np.load('data/X_test_test.npy')
    Y_test = np.load('data/Y_test_test.npy')
    X_private_test = np.load('data/X_private_test.npy')
    Y_private_test = np.load('data/Y_private_test.npy')
	 
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    print(X_private_test.shape)
    print(Y_private_test.shape)

   
    print("Training started...........")
    arrCallbacks = []
    epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print("epoch:", epoch))
    # save best weights having low validation loss
    checkpointer = ModelCheckpoint(filepath='my_model.h5', verbose=1, save_best_only=True)
    arrCallbacks.append(epoch_print_callback)
    arrCallbacks.append(checkpointer)
    
    BatchSize = 512
    hist =  model.fit(X_train, Y_train, epochs=30, batch_size=BatchSize, validation_data=(X_test, Y_test), shuffle=True, verbose=2, callbacks=arrCallbacks)
	
    print(hist.history.keys())
    train_accuracy = hist.history['acc']
    val_accuracy = hist.history['val_acc']
    
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    print ("Done!")
	
    #model accuracy both training and validation
    print("Train accuracy:")
    print(train_accuracy)
    print("Validation accuracy:")
    print(val_accuracy)
	
	# summarize by plotting graph history for accuracy
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
	
    # model loss both training and validation
    print("Train loss:")
    print(train_loss)
    print("Validation loss:")
    print(val_loss)

	
	# summarize by plotting graph history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # evaluation scores with loss and accuracy
    scores = model.evaluate(X_private_test, Y_private_test, verbose=0)
    print(scores)
    print("X_test loss :")
    print(scores[0])
    print("X_test accuracy :")
    print(scores[1])
    print("Training finished")"""
    
    
	
	
    
if __name__ == "__main__":
    main()