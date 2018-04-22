
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
import matplotlib.pyplot as plt
import itertools

def start_app(cnn):
    
    #load test dataset Usage-"PrivateTest"
     X_private_test = np.load('data/X_private_test.npy')
     Y_private_test = np.load('data/Y_private_test.npy')
     
     def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
            print(cm)
            
            
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, int(cm[i, j]*100)/100.0,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
     #Predict the features label   
     Y_pred_private = cnn.predict(X_private_test, batch_size=512, verbose=0)
     y_pred_private = np.argmax(Y_pred_private, axis=1)
     target_names = ['Angry','Disgust','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
     
     #print classification report
     print(classification_report(np.argmax(Y_private_test,axis=1), y_pred_private,target_names=target_names))
     
     
     #calculate confusion_matrix
     cnf_matrix = confusion_matrix(np.argmax(Y_private_test,axis=1),y_pred_private)
     
     #plot confusion_matrix
     plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=False,
                      title='Confusion Matrix for Test Dataset')
     plt.show()
     print("Prediction Finished")

        
if __name__ == '__main__':
   model = load_model('my_model.h5')
   print("Loaded model from disk")
   start_app(model)





