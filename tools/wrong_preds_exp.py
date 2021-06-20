import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def show_wrong_cases(predictions, true_classes, num_classes, images, one_hot = True,
                     show_count_plot = True, most_confused = True, k = 3):
    
    wrong_preds_inds = [list() for i in range(num_classes)]
    for i in range(len(predictions)):       
        if one_hot:
            if np.argmax(predictions[i]) != np.argmax(true_classes[i]):
                wrong_preds_inds[np.argmax(true_classes[i])].append(i)
        else:
            for p in range(len(predictions)):
                if predictions[p] > 0.5:
                    predictions[p] = 1
                else:
                    predictions[p] = 0
                    
            if predictions[i] != true_classes[i]:
                wrong_preds_inds[true_classes[i]].append(i)
#             print(predictions[i], true_classes[i])
    if show_count_plot:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0,0,1,1])
        x = [i for i in range(num_classes)]
        y = [len(w) for w in wrong_preds_inds]
        ax.bar(x, y)
        ax.set_title('false predicitons in each class')
        plt.show()
    
    if most_confused: # make it so that only wrong preds show
        if one_hot:
            LF = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        else:
            LF = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        losses = LF(true_classes, predictions).numpy()
        
        highest_loss_ind = losses.argsort()[-k:] #np.argmax(losses)
        
        losses.sort()
        highest_losses = losses[-k:]#np.max(losses)
        
        f, ax = plt.subplots(k, figsize=(4,20))
        for i in range(0, k):
            ax[i].imshow(images[highest_loss_ind[i]])
            ax[i].set_title('true class: '+str(np.argmax(true_classes[highest_loss_ind[i]])+1)
              +'/ predicted: '+ str(np.argmax(predictions[highest_loss_ind[i]])+1)+'/ idx: '+ str(highest_loss_ind[i]))
            ax[i].axis('off')
            ax[i].set_aspect('auto')
        plt.show()
    